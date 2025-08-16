-- =====================================================================
-- OPTIMIZED BUILDING-TO-ROAD CONNECTION PIPELINE
-- =====================================================================
-- This file contains two functions:
-- 1. setup_connection_infrastructure: Creates necessary indexes and a
--    sequence for primary keys. Run this once before the first execution.
-- 2. draw_building_connections: The optimized function to connect
--    buildings to the nearest road, splitting the road if necessary.
-- =====================================================================


-- =====================================================================
-- FUNCTION 1: setup_connection_infrastructure()
-- Description: Creates spatial indexes and a sequence for new way IDs.
-- Usage: SELECT setup_connection_infrastructure();
-- =====================================================================
CREATE OR REPLACE FUNCTION setup_connection_infrastructure() RETURNS void AS $$
BEGIN
    -- Drop existing indexes to prevent errors on re-run, not strictly necessary
    -- but good for idempotency.
    DROP INDEX IF EXISTS ways_tem_geom_idx;
    DROP INDEX IF EXISTS buildings_tem_center_idx;
    DROP SEQUENCE IF EXISTS ways_tem_way_id_seq;

    -- Create a sequence for unique, performant ID generation.
    -- This avoids the costly MAX(id) + 1 operation.
    CREATE SEQUENCE ways_tem_way_id_seq START 1;
    -- Set the sequence to start after the current maximum way_id to avoid conflicts.
    -- Using a separate transaction for this to ensure it completes before any potential locks.
    PERFORM setval('ways_tem_way_id_seq', (SELECT COALESCE(MAX(way_id), 0) + 1 FROM ways_tem), false);


    -- Create spatial GiST indexes. This is CRITICAL for performance.
    -- The planner will use these for ST_DWithin, ST_Intersects, and nearest neighbor (<->) searches.
    CREATE INDEX ways_tem_geom_idx ON ways_tem USING GIST (geom);
    CREATE INDEX buildings_tem_center_idx ON buildings_tem USING GIST (center);

    -- Analyze the tables to update statistics for the query planner.
    ANALYZE ways_tem;
    ANALYZE buildings_tem;

    RAISE NOTICE 'Infrastructure setup complete: Sequence created and spatial indexes are in place.';
END;
$$ LANGUAGE plpgsql;


-- =====================================================================
-- FUNCTION 2: draw_building_connections()
-- Description: Connects buildings to the road network in a single,
--              set-based operation.
-- Usage: SELECT draw_building_connections();
-- =====================================================================
CREATE OR REPLACE FUNCTION draw_building_connections() RETURNS void AS $$
DECLARE
    -- Define a small tolerance for snapping to existing vertices
    snap_tolerance double precision := 0.1;
BEGIN

    -- Use Common Table Expressions (CTEs) to perform all calculations before any modifications.
    -- This is a set-based approach, replacing the slow row-by-row loop.
    WITH
    -- Step 1: For each building, find the single closest road segment and calculate
    -- the connection point and line.
    nearest_roads AS (
        SELECT
            b.osm_id,
            b.center,
            road.way_id AS original_way_id,
            road.geom AS original_road_geom,
            -- ST_ClosestPoint is the point on the road line that is nearest to the building center.
            ST_ClosestPoint(road.geom, b.center) AS connection_point
        FROM
            buildings_tem b
        -- CROSS JOIN LATERAL allows us to run a "for each building" subquery efficiently.
        -- This is the modern, performant way to do a nearest neighbor search in PostGIS.
        CROSS JOIN LATERAL (
            SELECT
                w.way_id,
                w.geom
            FROM
                ways_tem w
            WHERE
                w.clazz != 110 -- Exclude existing connection lines
                AND ST_DWithin(b.center, w.geom, 2000) -- Use index to limit search radius
            ORDER BY
                w.geom <-> b.center -- Find the closest one
            LIMIT 1
        ) AS road
        WHERE b.peak_load_in_kw <> 0
    ),

    -- Step 2: Prepare all new geometries (the new connection line and the two parts of the split road)
    new_geometries AS (
        SELECT
            original_way_id,
            -- CASE statement handles snapping to existing start/end points if the connection
            -- point is within the defined tolerance. This avoids creating tiny, invalid segments.
            CASE
                WHEN ST_Distance(ST_StartPoint(original_road_geom), connection_point) < snap_tolerance
                    THEN ST_MakeLine(center, ST_StartPoint(original_road_geom))
                WHEN ST_Distance(ST_EndPoint(original_road_geom), connection_point) < snap_tolerance
                    THEN ST_MakeLine(center, ST_EndPoint(original_road_geom))
                ELSE ST_MakeLine(center, connection_point)
            END AS connection_line_geom,

            -- Determine if a split is needed. If the connection point is on a vertex, we don't split.
            CASE
                WHEN ST_Distance(ST_StartPoint(original_road_geom), connection_point) < snap_tolerance
                    OR ST_Distance(ST_EndPoint(original_road_geom), connection_point) < snap_tolerance
                THEN FALSE
                ELSE TRUE
            END AS needs_split,

            -- Calculate the first part of the split road
            ST_LineSubstring(original_road_geom, 0, ST_LineLocatePoint(original_road_geom, connection_point)) AS split_geom_1,
            -- Calculate the second part of the split road
            ST_LineSubstring(original_road_geom, ST_LineLocatePoint(original_road_geom, connection_point), 1) AS split_geom_2
        FROM
            nearest_roads
    ),

    -- Step 3: Delete the original roads that are being split. This is done in a single operation.
    deleted_ways AS (
        DELETE FROM ways_tem
        WHERE way_id IN (SELECT original_way_id FROM new_geometries WHERE needs_split = TRUE)
        RETURNING way_id
    )

    -- Step 4: Insert all the new geometries in a single batch operation.
    INSERT INTO ways_tem (way_id, clazz, geom, cost, reverse_cost)
    (
        -- Insert the new connection lines for all buildings
        SELECT
            nextval('ways_tem_way_id_seq'),
            110, -- clazz for building connection
            connection_line_geom,
            ST_Length(connection_line_geom),
            ST_Length(connection_line_geom)
        FROM
            new_geometries

        UNION ALL

        -- Insert the first half of the split roads
        SELECT
            nextval('ways_tem_way_id_seq'),
            103, -- clazz for a standard road segment
            split_geom_1,
            ST_Length(split_geom_1),
            ST_Length(split_geom_1)
        FROM
            new_geometries
        WHERE
            needs_split = TRUE AND ST_Length(split_geom_1) > 0.01 -- Avoid inserting zero-length lines

        UNION ALL

        -- Insert the second half of the split roads
        SELECT
            nextval('ways_tem_way_id_seq'),
            103, -- clazz for a standard road segment
            split_geom_2,
            ST_Length(split_geom_2),
            ST_Length(split_geom_2)
        FROM
            new_geometries
        WHERE
            needs_split = TRUE AND ST_Length(split_geom_2) > 0.01 -- Avoid inserting zero-length lines
    );

    RAISE NOTICE 'Building connections drawn successfully using a set-based approach.';
END;
$$ LANGUAGE plpgsql;
