--
-- Name: draw_home_connections(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE OR REPLACE FUNCTION draw_home_connections() RETURNS void
    LANGUAGE plpgsql
AS
$$
declare
    building   RECORD;
    new_line   RECORD;
    old_street RECORD;
begin
    -- Iterating buildings
    for building in
        SELECT osm_id, center
        FROM buildings_tem
        WHERE peak_load_in_kw <> 0
        loop
            --Finde Hausanschluss -> new_line
            SELECT building.osm_id as id,
                   110             as clazz,
                   a.line          as geom
            INTO new_line
            FROM (SELECT ST_ShortestLine(building.center, w.geom) as line
                  FROM ways_tem as w
                  WHERE w.clazz != 110 -- Hausanschlüsse und alte Straßen ausgeschlossen
                    AND ST_DWithin(building.center, w.geom, 2000)
                    AND ST_Distance(building.center, w.geom) > 0.1
                  ORDER BY building.center <-> w.geom
                  LIMIT 1) as a;

            IF NOT FOUND THEN
                continue;
            END IF;
            -- Finde die angeschlossene Straße -> old_street
            SELECT *,
                   ST_LineInterpolatePoint(ST_Intersection(
                                                   ST_Buffer(new_line.geom, 0.1), w.geom), 0.5) as connection_point
            INTO old_street
            FROM ways_tem as w
            WHERE ST_DWithin(new_line.geom, w.geom, 1000)                -- begrenzen
              AND ST_Intersects(ST_Buffer(new_line.geom, 0.001), w.geom) -- Kontakt existiert
              AND w.clazz != 110
              AND ST_GeometryType(ST_Intersection(
                    ST_Buffer(new_line.geom, 0.1), w.geom)) = 'ST_LineString'
            ORDER BY new_line.geom <-> w.geom
            LIMIT 1; -- zur Garantie gegen mehrere Kontakte

            IF NOT FOUND THEN
                continue;
            END IF;
            -- Überprüfe ob das Lot (ShortestLine) nah zu einem Knoten ist, wenn ja verschiebe
            IF ST_Distance(ST_StartPoint(old_street.geom), old_street.connection_point) < 0.1 THEN

                new_line.geom := ST_Makeline(building.center, ST_StartPoint(old_street.geom));
                INSERT INTO ways_tem (way_id, clazz, geom)
                SELECT Max(way_id) + 1,
                       new_line.clazz,
                       new_line.geom
                FROM ways_tem;
                --raise notice '<0.01';
                continue;
            ELSEIF ST_Distance(ST_EndPoint(old_street.geom), old_street.connection_point) < 0.1 THEN

                new_line.geom := ST_Makeline(building.center, ST_EndPoint(old_street.geom));
                INSERT INTO ways_tem (way_id, clazz, geom)
                SELECT Max(way_id) + 1,
                       new_line.clazz,
                       new_line.geom
                FROM ways_tem;
                --raise notice '>0.99';
                continue;
            ELSE
                INSERT INTO ways_tem (way_id, clazz, geom)
                SELECT Max(way_id) + 1,
                       new_line.clazz,
                       new_line.geom
                FROM ways_tem;
                --raise notice 'normal';
            END IF;

            --Hausanschluss schneidet die Straße
            --old_street clazz ungültig machen
            DELETE
            FROM ways_tem
            WHERE way_id = old_street.way_id;

            -- INSERT new streets as two part of old street
            -- 	first half
            INSERT INTO ways_tem (way_id, clazz, geom)
            SELECT Max(way_id) + 1, --unique id guarenteed
                   103,
                   ST_LineSubstring(old_street.geom, 0, ST_LineLocatePoint(
                           old_street.geom, old_street.connection_point))
            FROM ways_tem;
            --  second half
            INSERT INTO ways_tem (way_id, clazz, geom)
            SELECT Max(way_id) + 1,
                   103,
                   ST_LineSubstring(old_street.geom, ST_LineLocatePoint(
                           old_street.geom, old_street.connection_point), 1)
            FROM ways_tem;


        end loop;
    raise notice 'Home connections are drawn successfully...';
end;
$$;



--
-- Name: draw_way_connections(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE OR REPLACE FUNCTION draw_way_connections() RETURNS void
    LANGUAGE PLPGSQL
AS
$$
declare
    way               RECORD;
    interpolate_point RECORD;
    old_street        RECORD;
begin
    -- Iterating buildings
    for way in
        SELECT geom, clazz, way_id
        FROM ways_tem
        loop
            --Finde Hausanschluss -> new_line
            SELECT geom, clazz, way_id
            INTO old_street
            FROM ways_tem as w
            WHERE ST_Intersects(ST_LineSubstring(way.geom, 0.01, 0.99), w.geom) -- begrenzen
              AND w.way_id != way.way_id
            LIMIT 1;

            IF NOT FOUND THEN
                continue;
            END IF;

            -- check whether the intersection of ST_Buffer(way.geom,0.1) , old_street.geom is a line
            -- this is necessary for the next SELECT statement with ST_LineInterpolatePoint()-function
            IF ST_Geometrytype(ST_Intersection(ST_Buffer(way.geom, 0.1), old_street.geom)) != 'ST_LineString' THEN
                RAISE NOTICE 'Value: %', old_street.geom;
                continue;
            END IF;

            SELECT ST_LineInterpolatePoint(ST_Intersection(ST_Buffer(way.geom, 0.1), old_street.geom), 0.5) AS geom
            INTO interpolate_point;

            DELETE
            FROM ways_tem
            WHERE way_id = old_street.way_id;

            -- INSERT new streets as two part of old street
            -- 	first half
            INSERT INTO ways_tem (way_id, clazz, geom)
            SELECT Max(way_id) + 1, --unique id guarenteed
                   old_street.clazz,
                   ST_LineSubstring(old_street.geom, 0, ST_LineLocatePoint(
                           old_street.geom, interpolate_point.geom))
            FROM ways_tem;
            --  second half
            INSERT INTO ways_tem (way_id, clazz, geom)
            SELECT Max(way_id) + 1,
                   old_street.clazz,
                   ST_LineSubstring(old_street.geom, ST_LineLocatePoint(
                           old_street.geom, interpolate_point.geom), 1)
            FROM ways_tem;

            DELETE
            FROM ways_tem
            WHERE way_id = way.way_id;

            INSERT INTO ways_tem (way_id, clazz, geom)
            SELECT Max(way_id) + 1, --unique id guarenteed
                   way.clazz,
                   ST_LineSubstring(way.geom, 0, ST_LineLocatePoint(
                           way.geom, interpolate_point.geom))
            FROM ways_tem;
            --  second half
            INSERT INTO ways_tem (way_id, clazz, geom)
            SELECT Max(way_id) + 1,
                   way.clazz,
                   ST_LineSubstring(way.geom, ST_LineLocatePoint(
                           way.geom, interpolate_point.geom), 1)
            FROM ways_tem;

        end loop;
    raise notice 'Home connections are drawn successfully...';
end;
$$;