
-- ==============================================
-- SETUP FUNCTION: Configure temp tables after creation
-- ==============================================
CREATE OR REPLACE FUNCTION setup_temp_tables()
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    -- Setup ways_tem table
    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = 'ways_tem_way_id_seq'
    ) THEN
        CREATE SEQUENCE ways_tem_way_id_seq;
        PERFORM setval('ways_tem_way_id_seq', COALESCE((SELECT MAX(way_id) FROM ways_tem), 0) + 1);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'ways_tem' 
        AND column_name = 'way_id' 
        AND column_default IS NOT NULL
    ) THEN
        ALTER TABLE ways_tem
            ALTER COLUMN way_id
            SET DEFAULT nextval('ways_tem_way_id_seq');
    END IF;
    
    -- Create spatial indexes for both temp tables
    CREATE INDEX IF NOT EXISTS idx_ways_geom      ON ways_tem     USING gist (geom);
    CREATE INDEX IF NOT EXISTS idx_buildings_geom ON buildings_tem USING gist (center);    
    -- Update statistics
    ANALYZE ways_tem;
    ANALYZE buildings_tem;
    END;
$$;

CREATE OR REPLACE FUNCTION draw_home_connections_set_based()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    t0  TIMESTAMP := clock_timestamp();
BEGIN
    -- Create temporary table to store connection data
    CREATE TEMP TABLE IF NOT EXISTS temp_connections AS
    WITH building_connections AS (
        SELECT  b.osm_id,
                b.center AS building_center,
                w.way_id AS street_id,
                w.geom AS street_geom,
                w.clazz AS street_clazz,
                ST_ShortestLine(b.center, w.geom) AS drop_line,
                -- Calculate EXACT connection point where drop intersects street
                ST_LineInterpolatePoint(
                    ST_Intersection(ST_Buffer(ST_ShortestLine(b.center, w.geom), 0.1), w.geom), 
                    0.5
                ) AS exact_connection_point,
                row_number() OVER (PARTITION BY b.osm_id ORDER BY b.center <-> w.geom) AS rn
        FROM   buildings_tem b
        CROSS  JOIN LATERAL (
                 SELECT w.*
                 FROM   ways_tem w
                 WHERE  w.clazz <> 110                -- ignore existing drops
                   AND ST_DWithin(b.center, w.geom, 2000)
                   AND ST_Distance(b.center, w.geom) > 0.1
                   -- Ensure intersection exists and is valid
                   AND ST_Intersects(ST_Buffer(ST_ShortestLine(b.center, w.geom), 0.001), w.geom)
                   AND ST_GeometryType(ST_Intersection(ST_Buffer(ST_ShortestLine(b.center, w.geom), 0.1), w.geom)) = 'ST_LineString'
                 ORDER  BY b.center <-> w.geom
                 LIMIT  1
               ) AS w
        WHERE  b.peak_load_in_kw > 0
    )
    SELECT  osm_id,
            building_center,
            street_id,
            street_geom,
            street_clazz,
            drop_line,
            exact_connection_point,
            -- Determine connection type based on proximity to endpoints
            CASE 
                WHEN ST_Distance(ST_StartPoint(street_geom), exact_connection_point) < 0.1 THEN 'start'
                WHEN ST_Distance(ST_EndPoint(street_geom), exact_connection_point) < 0.1 THEN 'end'
                ELSE 'split'
            END AS connection_type,
            -- Create final drop geometry using EXACT connection point
            CASE 
                WHEN ST_Distance(ST_StartPoint(street_geom), exact_connection_point) < 0.1 THEN 
                    ST_MakeLine(building_center, ST_StartPoint(street_geom))
                WHEN ST_Distance(ST_EndPoint(street_geom), exact_connection_point) < 0.1 THEN 
                    ST_MakeLine(building_center, ST_EndPoint(street_geom))
                ELSE ST_MakeLine(building_center, exact_connection_point)  -- Use exact point!
            END AS final_drop_geom,
            ST_LineLocatePoint(street_geom, exact_connection_point) AS split_fraction
    FROM building_connections
    WHERE rn = 1;

    ------------------------------------------------------------------
    -- Step 1: Insert ALL service drops first (using exact connection points)
    ------------------------------------------------------------------
    INSERT INTO ways_tem (way_id, clazz, geom)
    SELECT nextval('ways_tem_way_id_seq'), 110, final_drop_geom
    FROM temp_connections;

    ------------------------------------------------------------------
    -- Step 2: Split streets that need splitting (maintaining exact points)
    ------------------------------------------------------------------
    WITH streets_to_split AS (
        SELECT  street_id,
                street_geom,
                street_clazz,
                exact_connection_point,
                split_fraction
        FROM   temp_connections
        WHERE  connection_type = 'split'
          AND  split_fraction > 0.01         -- Avoid splitting too close to start
          AND  split_fraction < 0.99         -- Avoid splitting too close to end
    ),
    street_segments AS (
        SELECT  street_id,
                street_clazz,
                exact_connection_point,
                -- First segment: start to EXACT connection point
                ST_LineSubstring(street_geom, 0, split_fraction) AS segment1,
                -- Second segment: EXACT connection point to end  
                ST_LineSubstring(street_geom, split_fraction, 1) AS segment2
        FROM   streets_to_split
    ),
    deleted_streets AS (
        DELETE FROM ways_tem w
        USING streets_to_split s
        WHERE w.way_id = s.street_id
        RETURNING 1
    )
    -- Insert new street segments (both ending/starting at exact connection point)
    INSERT INTO ways_tem (way_id, clazz, geom)
    SELECT nextval('ways_tem_way_id_seq'), street_clazz, segment1
    FROM   street_segments
    WHERE  ST_Length(segment1) > 0.01
    UNION ALL
    SELECT nextval('ways_tem_way_id_seq'), street_clazz, segment2
    FROM   street_segments  
    WHERE  ST_Length(segment2) > 0.01;

    -- Clean up temp table
    DROP TABLE IF EXISTS temp_connections;

    RAISE NOTICE 'Home connections finished in %.2f s (topology-preserving set-based)',
          EXTRACT(EPOCH FROM clock_timestamp() - t0);
END;
$$;
