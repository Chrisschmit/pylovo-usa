# Database schema - table structure
CREATE_QUERIES = {
    "res": """CREATE TABLE IF NOT EXISTS res (
        osm_id varchar PRIMARY KEY,
        area numeric(23, 15),
        use varchar(80),
        comment varchar(80),
        free_walls integer,
        building_t varchar(80),
        occupants numeric(23, 15),
        floors integer,
        constructi varchar(80),
        refurb_wal numeric(23, 15),
        refurb_roo numeric(23, 15),
        refurb_bas numeric(23, 15),
        refurb_win numeric(23, 15),
        geom geometry(MultiPolygon,%(epsg)s)
    )
    """,
    "oth": """CREATE TABLE IF NOT EXISTS oth (
        osm_id varchar PRIMARY KEY,
        area numeric(23, 15),
        use varchar(80),
        comment varchar(80),
        free_walls integer,
        geom geometry(MultiPolygon,%(epsg)s)
    )
    """,
    "equipment_data": """
    CREATE TABLE IF NOT EXISTS equipment_data (
        name varchar(100) PRIMARY KEY,
        s_max_kva integer,
        max_i_a integer,
        r_mohm_per_km integer,
        x_mohm_per_km integer,
        z_mohm_per_km integer,
        cost_eur integer,
        typ varchar(50),
        application_area integer
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
    "classification_version": """
    CREATE TABLE IF NOT EXISTS classification_version (
        classification_id integer NOT NULL,
        version_comment varchar, 
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        classification_region varchar,
        CONSTRAINT classification_pkey PRIMARY KEY (classification_id)
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
        plz bigint UNIQUE NOT NULL,
        qkm double precision,
        population integer,
        geom geometry(MultiPolygon,%(epsg)s)
    )
    """,
    "postcode_result": """
    CREATE TABLE IF NOT EXISTS postcode_result (   
        version_id varchar(10) NOT NULL,
        postcode_result_plz bigint NOT NULL,
        settlement_type integer,
        geom geometry(MultiPolygon,%(epsg)s),
        house_distance numeric,
        CONSTRAINT "postcode_result_pkey" PRIMARY KEY (version_id, postcode_result_plz),
        CONSTRAINT fk_postcode_result_version_id
            FOREIGN KEY (version_id)
            REFERENCES version (version_id)
            ON DELETE CASCADE,
        CONSTRAINT fk_postcode_result_plz
            FOREIGN KEY (postcode_result_plz)
            REFERENCES postcode (plz)
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
        plz bigint NOT NULL,
        transformer_rated_power bigint,
        model_status integer,
        ont_vertice_id bigint,
        grid json,
        CONSTRAINT cluster_identifier UNIQUE (version_id, kcid, bcid, plz),
        CONSTRAINT unique_grid_result_id_version_id UNIQUE (version_id, grid_result_id),
        CONSTRAINT fk_grid_result_version_id_plz
            FOREIGN KEY (version_id, plz)
            REFERENCES postcode_result (version_id, postcode_result_plz)
            ON DELETE CASCADE
    );
    CREATE INDEX idx_grid_result_version_id_plz_bcid_kcid
    ON grid_result (version_id, plz, bcid, kcid)
    """,
    "lines_result": """
    CREATE TABLE IF NOT EXISTS lines_result (
        lines_result_id SERIAL PRIMARY KEY,
        grid_result_id bigint NOT NULL,
        geom geometry(LineString,%(epsg)s),
        line_name varchar(15),
        std_type varchar(15),
        from_bus integer,
        to_bus integer,
        length_km numeric,
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
        geom geometry(MultiPolygon,%(epsg)s),
        houses_per_building integer,
        center geometry(Point,%(epsg)s),
        peak_load_in_kw numeric,
        vertice_id integer,
        floors integer,
        connection_point integer,
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
    "municipal_register": """CREATE TABLE IF NOT EXISTS municipal_register (
        plz bigint,
        pop bigint,
        area double precision,
        lat double precision,
        lon double precision,
        ags bigint,
        name_city text,
        fed_state integer,
        regio7 integer,
        regio5 integer,
        pop_den double precision,
        CONSTRAINT municipal_register_pkey PRIMARY KEY (plz, ags)
    )
    """,
    "sample_set": """CREATE TABLE IF NOT EXISTS sample_set (
        classification_id integer NOT NULL,
        plz bigint NOT NULL,
        ags bigint,
        bin_no int,
        bins numeric,
        perc_bin numeric,
        count numeric,
        perc numeric,
        CONSTRAINT sample_set_pkey PRIMARY KEY (classification_id, plz),
        CONSTRAINT fk_sample_set_classification_id
            FOREIGN KEY (classification_id)
            REFERENCES classification_version (classification_id)
            ON DELETE CASCADE,
        CONSTRAINT fk_sample_set_plz
            FOREIGN KEY (plz, ags)
            REFERENCES municipal_register (plz, ags)
            ON DELETE CASCADE
    )
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
        geom geometry(Point,%(epsg)s),
        osm_id varchar,
        version_id varchar(10),
        "comment" varchar,
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


    "transformer_classified": """
    CREATE TABLE IF NOT EXISTS transformer_classified (
        grid_result_id bigint NOT NULL,
        geom geometry(Point,%(epsg)s),
        kmedoid_clusters integer,
        kmedoid_representative_grid bool,
        kmeans_clusters integer,
        kmeans_representative_grid bool,
        gmm_clusters integer,
        gmm_representative_grid bool,
        classification_id integer NOT NULL,
        CONSTRAINT pk_grid_result_id PRIMARY KEY (grid_result_id, classification_id),
        CONSTRAINT fk_transformer_classified_classification_id
            FOREIGN KEY (classification_id)
            REFERENCES classification_version (classification_id)
            ON DELETE CASCADE,
        CONSTRAINT fk_transformer_classified_grid_result
            FOREIGN KEY (grid_result_id)
            REFERENCES grid_result (grid_result_id)
            ON DELETE CASCADE
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
        geom geometry(LineString,%(epsg)s),
        way_id integer PRIMARY KEY
    )
    """,
    "ways_result": """
    CREATE TABLE IF NOT EXISTS ways_result (
        version_id varchar(10) NOT NULL,
        clazz integer,
        source integer,
        target integer,
        cost double precision,
        reverse_cost double precision,
        geom geometry(LineString,%(epsg)s),
        way_id integer NOT NULL,
        plz bigint,
        CONSTRAINT pk_ways_result PRIMARY KEY (version_id, way_id, plz),
        CONSTRAINT fk_ways_result_version_id_plz
            FOREIGN KEY (version_id, plz)
            REFERENCES postcode_result (version_id, postcode_result_plz)
            ON DELETE CASCADE
    )
    """,
    # old name: grid_parameters
    # saves grid parameters for a whole plz for visualization
    "plz_parameters": """
    CREATE TABLE IF NOT EXISTS plz_parameters (
        version_id varchar(10) NOT NULL,
        plz bigint NOT NULL,
        trafo_num json,
        cable_length json,
        load_count_per_trafo json,
        bus_count_per_trafo json,
        sim_peak_load_per_trafo json,
        max_distance_per_trafo json,
        avg_distance_per_trafo json,
        CONSTRAINT parameters_pkey PRIMARY KEY (version_id, plz),
        CONSTRAINT fk_plz_parameters_version_id_plz
            FOREIGN KEY (version_id, plz)
            REFERENCES postcode_result (version_id, postcode_result_plz)
            ON DELETE CASCADE
    )
    """,
    "transformer_positions_with_grid": """
    CREATE OR REPLACE VIEW transformer_positions_with_grid AS (
        SELECT tp.*, gr.kcid, gr.bcid, gr.plz
        FROM transformer_positions tp
        JOIN grid_result gr ON tp.grid_result_id = gr.grid_result_id
    )
    """,
    "transformer_classified_with_grid": """
    CREATE OR REPLACE VIEW transformer_classified_with_grid AS (
        SELECT tc.*, gr.version_id, gr.kcid, gr.bcid, gr.plz
        FROM transformer_classified tc
        JOIN grid_result gr ON tc.grid_result_id = gr.grid_result_id
    )
    """,
    "buildings_result_with_grid": """
    CREATE OR REPLACE VIEW buildings_result_with_grid AS (
        SELECT
            (br.version_id || '_' || br.osm_id) AS id,
            br.*,
            gr.kcid, gr.bcid, gr.plz
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
            gr.version_id, gr.kcid, gr.bcid, gr.plz
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
        geom geometry(Geometry,%(epsg)s),  -- needs to be geometry as multipoint & multipolygon get inserted here
        houses_per_building integer,
        center geometry(Point,%(epsg)s),
        peak_load_in_kw numeric,
        plz bigint,
        vertice_id bigint,
        bcid integer,
        kcid integer,
        floors integer,
        connection_point integer
    )""",
    "ways_tem": """CREATE TABLE IF NOT EXISTS ways_tem
    (
        clazz integer,
        source integer,
        target integer,
        cost double precision,
        reverse_cost double precision,
        geom geometry(LineString,%(epsg)s),
        way_id integer,
        plz bigint
    )""",
}