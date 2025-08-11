# Database schema - table structure
CREATE_QUERIES = {
    "res": """CREATE TABLE IF NOT EXISTS res (
        osm_id varchar PRIMARY KEY,
        c_block_id bigint,
        build_id bigint,
        area numeric(23, 15),
        use varchar(80),
        comment varchar(80),
        free_walls integer,
        build_type varchar(80),
        occupants integer,
        height float,
        housing_un float,
        floors integer,
        constructi varchar(80),
        refurb_wal float,
        refurb_roo float,
        refurb_bas float,
        refurb_win float,
        geom geometry(MultiPolygon,%(epsg)s)
    )
    """,
    "oth": """CREATE TABLE IF NOT EXISTS oth (
        osm_id varchar PRIMARY KEY,
        c_block_id bigint,
        build_id bigint,
        area numeric(23, 15),
        use varchar(80),
        height float,
        comment varchar(80),
        free_walls integer,
        geom geometry(MultiPolygon,%(epsg)s)
    )
    """,
    "equipment_data":
    """CREATE TABLE IF NOT EXISTS equipment_data (
    name VARCHAR(100) PRIMARY KEY,           -- Equipment name
    type VARCHAR(50),                        -- Equipment category: 'Substation', 'Transformer', 'Line'
    application_area INTEGER,                -- Defines to which settlement type the equipment belongs to
    ovh_ung VARCHAR(12),                     -- Installation type: 'Overhead' or 'Underground'
    n_phases SMALLINT,                       -- Number of phases (1, 2, or 3)
    voltage_level VARCHAR(25),               -- Nominal voltage class (e.g.'HV-MV', 'MV-LV')
    s_max_kva INTEGER,                       -- Rated apparent power in kVA
    primary_voltage_kv NUMERIC,              -- Primary side voltage in kilovolts
    secondary_voltage_kv NUMERIC,            -- Secondary side voltage in kilovolts
    reactance_pu NUMERIC,                    -- Per-unit reactance
    no_load_losses_kw NUMERIC,               -- No-load (core) losses in kilowatts
    short_circuit_res_ohm NUMERIC,           -- Short-circuit resistance in ohms
    r_ohm_per_km NUMERIC,                    -- Resistance per km in ohms 
    x_ohm_per_km NUMERIC,                    -- Inductive reactance per km in ohms 
    z_ohm_per_km NUMERIC,                    -- Impedance per km in ohms
    capacitance_nf_per_km NUMERIC,           -- Capacitance per km in nanofarads
    max_i_a INTEGER,                         -- Maximum current in amperes
    line_voltage NUMERIC,                    -- Nominal line voltage in kilovolts
    cost NUMERIC                             -- Investment cost (currency assumed unless extra field added)
    )
    """,
    "version": """CREATE TABLE IF NOT EXISTS version (
        version_id varchar(10) PRIMARY KEY,
        version_comment varchar, 
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        consumer_categories varchar,
        cable_cost_dict varchar,
        connection_available_cables varchar,   
        other_parameters varchar
    )
    """,
    "postcode": """
    CREATE TABLE IF NOT EXISTS postcode (
        id SERIAL PRIMARY KEY,
        state_abbr varchar,
        state_fips varchar,
        county_fips varchar,
        county_name varchar,
        subdivision_fips varchar,
        subdivision_name varchar,
        funcstat varchar,
        regional_identifier bigint UNIQUE NOT NULL,
        qkm double precision,
        population integer,
        geom geometry(MultiPolygon,%(epsg)s)
    )
    """,
    "postcode_result": """
    CREATE TABLE IF NOT EXISTS postcode_result (   
        version_id varchar(10) NOT NULL,
        postcode_result_regional_identifier bigint NOT NULL,
        settlement_type integer,
        load_density numeric,
        geom geometry(MultiPolygon,%(epsg)s),
        CONSTRAINT "postcode_result_pkey" PRIMARY KEY (version_id, postcode_result_regional_identifier),
        CONSTRAINT fk_postcode_result_version_id
            FOREIGN KEY (version_id)
            REFERENCES version (version_id)
            ON DELETE CASCADE,
        CONSTRAINT fk_postcode_result_regional_identifier
            FOREIGN KEY (postcode_result_regional_identifier)
            REFERENCES postcode (regional_identifier)
            ON DELETE CASCADE
    )
    """,
    # old name: building_clusters, got merged with grids
    "grid_result": """
    CREATE TABLE IF NOT EXISTS grid_result (
        grid_result_id SERIAL PRIMARY KEY,
        version_id varchar(10) NOT NULL,
        kcid integer NOT NULL,
        bcid integer NOT NULL,
        regional_identifier bigint NOT NULL,
        transformer_rated_power bigint,
        model_status integer,
        transformer_vertice_id bigint,
        grid json,
        CONSTRAINT cluster_identifier UNIQUE (version_id, kcid, bcid, regional_identifier),
        CONSTRAINT unique_grid_result_id_version_id UNIQUE (version_id, grid_result_id),
        CONSTRAINT fk_grid_result_version_id_regional_identifier
            FOREIGN KEY (version_id, regional_identifier)
            REFERENCES postcode_result (version_id, postcode_result_regional_identifier)
            ON DELETE CASCADE
    );
    CREATE INDEX idx_grid_result_version_id_regional_identifier_bcid_kcid
    ON grid_result (version_id, regional_identifier, bcid, kcid)
    """,
    "lines_result": """
    CREATE TABLE IF NOT EXISTS lines_result (
        lines_result_id SERIAL PRIMARY KEY,
        grid_result_id bigint NOT NULL,
        line_name varchar(15),
        std_type varchar(15),
        from_bus integer,
        to_bus integer,
        length_km numeric,
        geom geometry(LineString,%(epsg)s),
        CONSTRAINT fk_lines_result_grid_result
            FOREIGN KEY (grid_result_id)
            REFERENCES grid_result (grid_result_id)
            ON DELETE CASCADE
    )
    """,
    "consumer_categories": """
    CREATE TABLE IF NOT EXISTS consumer_categories (
        consumer_category_id integer PRIMARY KEY,
        definition varchar(30) UNIQUE NOT NULL,
        peak_load numeric(10,2),
        yearly_consumption numeric(10,2),
        peak_load_per_m2 numeric(10,2),
        yearly_consumption_per_m2 numeric(10,2),
        sim_factor numeric(10,2) NOT NULL
    )
    """,
    "buildings_result": """
    CREATE TABLE IF NOT EXISTS buildings_result (
        version_id varchar(10) NOT NULL,
        osm_id varchar NOT NULL,
        grid_result_id bigint NOT NULL,
        area numeric,
        type varchar(30),
        houses_per_building integer,
        peak_load_in_kw numeric,
        vertice_id integer,
        floors integer,
        connection_point integer,
        center geometry(Point,%(epsg)s),
        geom geometry(MultiPolygon,%(epsg)s),
        CONSTRAINT buildings_result_pkey PRIMARY KEY (version_id, osm_id),
        CONSTRAINT fk_buildings_result_grid_result
            FOREIGN KEY (version_id, grid_result_id)
            REFERENCES grid_result (version_id, grid_result_id)
            ON DELETE CASCADE,
        CONSTRAINT fk_buildings_result_type
            FOREIGN KEY (type)
            REFERENCES consumer_categories (definition)
            ON DELETE CASCADE
    );
    CREATE INDEX idx_buildings_result_grid_result_id
    ON buildings_result (grid_result_id);
    """,
    "clustering_parameters": """CREATE TABLE IF NOT EXISTS clustering_parameters (
        grid_result_id bigint PRIMARY KEY,
        
        no_connection_buses integer,
        no_branches integer,
        
        no_house_connections integer,
        no_house_connections_per_branch numeric,
        no_households integer,
        no_household_equ numeric,
        no_households_per_branch numeric,
        max_no_of_households_of_a_branch numeric,
        house_distance_km numeric,
        
        transformer_mva numeric,
        osm_trafo bool,
        
        max_trafo_dis numeric,
        avg_trafo_dis numeric,
        
        cable_length_km numeric,
        cable_len_per_house numeric,
        
        max_power_mw numeric,
        simultaneous_peak_load_mw numeric,
        
        resistance numeric,
        reactance numeric,
        ratio numeric,
        vsw_per_branch numeric,
        max_vsw_of_a_branch numeric,
        
        filtered boolean,
        CONSTRAINT fk_clustering_parameters_grid_result
            FOREIGN KEY (grid_result_id)
            REFERENCES grid_result (grid_result_id)
            ON DELETE CASCADE
    )
    """,
    "transformers": """CREATE TABLE IF NOT EXISTS transformers (
        osm_id varchar PRIMARY KEY,
        area double precision,
        power varchar,
        geom_type varchar,
        within_shopping boolean,
        geom geometry(MultiPoint, %(epsg)s)
    )
    """,
    
    "transformer_positions": """
    CREATE TABLE IF NOT EXISTS transformer_positions (
        grid_result_id bigint PRIMARY KEY,
        osm_id varchar,
        version_id varchar(10),
        "comment" varchar,
        geom geometry(Point,%(epsg)s),

        CONSTRAINT uq_tp_osm_v UNIQUE (osm_id, version_id),
        CONSTRAINT fk_tp_version_id
            FOREIGN KEY (version_id)
            REFERENCES version (version_id)
            ON DELETE CASCADE,
        CONSTRAINT fk_tp_grid_result_id
            FOREIGN KEY (grid_result_id)
            REFERENCES grid_result (grid_result_id)
            ON DELETE CASCADE,
        CONSTRAINT fk_tp_osm_id
            FOREIGN KEY (osm_id)
            REFERENCES transformers (osm_id)
    )
    """,


    "fips_log": """
    CREATE TABLE IF NOT EXISTS fips_log (
        fips_code bigint PRIMARY KEY
    )
    """,
    "ways": """
    CREATE TABLE IF NOT EXISTS ways (
        clazz integer,
        source integer,
        target integer,
        cost double precision,
        reverse_cost double precision,
        way_id integer PRIMARY KEY,
        geom geometry(LineString,%(epsg)s)
    )
    """,
    "ways_result": """
    CREATE TABLE IF NOT EXISTS ways_result (
        version_id varchar(10) NOT NULL,
        way_id integer NOT NULL,
        regional_identifier bigint,
        clazz integer,
        source integer,
        target integer,
        cost double precision,
        reverse_cost double precision,

        geom geometry(LineString,%(epsg)s),
        CONSTRAINT pk_ways_result PRIMARY KEY (version_id, way_id, regional_identifier),
        CONSTRAINT fk_ways_result_version_id_regional_identifier
            FOREIGN KEY (version_id, regional_identifier)
            REFERENCES postcode_result (version_id, postcode_result_regional_identifier)
            ON DELETE CASCADE
    )
    """,
    # old name: grid_parameters
    # saves grid parameters for a whole regional_identifier for visualization
    "regional_identifier_parameters": """
    CREATE TABLE IF NOT EXISTS regional_identifier_parameters (
        version_id varchar(10) NOT NULL,
        regional_identifier bigint NOT NULL,
        trafo_num json,
        cable_length json,
        load_count_per_trafo json,
        bus_count_per_trafo json,
        sim_peak_load_per_trafo json,
        max_distance_per_trafo json,
        avg_distance_per_trafo json,
        CONSTRAINT parameters_pkey PRIMARY KEY (version_id, regional_identifier),
        CONSTRAINT fk_regional_identifier_parameters_version_id_regional_identifier
            FOREIGN KEY (version_id, regional_identifier)
            REFERENCES postcode_result (version_id, postcode_result_regional_identifier)
            ON DELETE CASCADE
    )
    """,
    "transformer_positions_with_grid": """
    CREATE OR REPLACE VIEW transformer_positions_with_grid AS (
        SELECT tp.*, gr.kcid, gr.bcid, gr.regional_identifier
        FROM transformer_positions tp
        JOIN grid_result gr ON tp.grid_result_id = gr.grid_result_id
    )
    """,
    "buildings_result_with_grid": """
    CREATE OR REPLACE VIEW buildings_result_with_grid AS (
        SELECT
            (br.version_id || '_' || br.osm_id) AS id,
            br.*,
            gr.kcid, gr.bcid, gr.regional_identifier
        FROM buildings_result br
        JOIN grid_result gr ON br.grid_result_id = gr.grid_result_id
    )
    """,
    "lines_result_with_grid": """
    CREATE OR REPLACE VIEW lines_result_with_grid AS (
        SELECT
            lr.lines_result_id as id,
            lr.grid_result_id,
            lr.geom,
            lr.line_name,
            lr.std_type,
            lr.from_bus,
            lr.to_bus,
            lr.length_km,
            gr.version_id, gr.kcid, gr.bcid, gr.regional_identifier
        FROM lines_result lr
        JOIN grid_result gr ON lr.grid_result_id = gr.grid_result_id
    )
    """
}

TEMP_CREATE_QUERIES = {
    "buildings_tem": """CREATE TABLE IF NOT EXISTS buildings_tem
    (
        osm_id varchar,
        area numeric,
        type varchar(80),
        houses_per_building integer,
        peak_load_in_kw numeric,
        regional_identifier bigint,
        vertice_id bigint,
        bcid integer,
        kcid integer,
        floors integer,
        connection_point integer,center geometry(Point,%(epsg)s),
        geom geometry(Geometry,%(epsg)s)  -- needs to be geometry as multipoint & multipolygon get inserted here

    )""",
    "ways_tem": """CREATE TABLE IF NOT EXISTS ways_tem
    (
        clazz integer,
        source integer,
        target integer,
        cost double precision,
        reverse_cost double precision,
        way_id integer,
        regional_identifier bigint,
        geom geometry(LineString,%(epsg)s)

    )""",
}
