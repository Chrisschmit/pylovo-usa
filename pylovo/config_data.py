import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# Database connection configuration
DBNAME = os.getenv("DBNAME", "pylovo_db_local")
USER = os.getenv("USER", "postgres")
HOST = os.getenv("HOST", "localhost")
PORT = os.getenv("PORT", "5432")
PASSWORD = os.getenv("PASSWORD", "postgres")

# Directory where the result csv and json files be saved
RESULT_DIR = os.path.join(os.getcwd(), "results")

# Toggles whether the grid json files will be saved in a folder or just in the db
SAVE_GRID_FOLDER = False

# Logging configuration for PgReaderWriter & GridGenerator
LOG_LEVEL = "INFO"

CSV_FILE_LIST = [
    {"path": os.path.join("raw_data", "equipment_data.csv"), "table_name": "equipment_data"},
    {"path": os.path.join("raw_data", "postcode.csv"), "table_name": "postcode"},
]

CLUSTERING_PARAMETERS = ["version_id",
                         "plz",
                         "bcid",
                         "kcid",
                         "no_connection_buses",
                         "no_branches",
                         "no_house_connections",
                         "no_house_connections_per_branch",
                         "no_households",
                         "no_household_equ",
                         "no_households_per_branch",
                         "max_no_of_households_of_a_branch",
                         "house_distance_km",
                         "transformer_mva",
                         "osm_trafo",
                         "max_trafo_dis",
                         "avg_trafo_dis",
                         "cable_length_km",
                         "cable_len_per_house",
                         "max_power_mw",
                         "simultaneous_peak_load_mw",
                         "resistance",
                         "reactance",
                         "ratio",
                         "vsw_per_branch",
                         "max_vsw_of_a_branch"]

MUNICIPAL_REGISTER = ['plz', 'pop', 'area', 'lat', 'lon', 'ags', 'name_city', 'fed_state', 'regio7', 'regio5',
                      'pop_den']

# Database schema - table structure
CREATE_QUERIES = {
    "res": """CREATE TABLE IF NOT EXISTS public.res
(
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
    geom geometry(MultiPolygon,3035)
)""",
    "oth": """CREATE TABLE IF NOT EXISTS public.oth
(
    osm_id varchar PRIMARY KEY,
    area numeric(23, 15),
    use varchar(80),
    comment varchar(80),
    free_walls integer,
    geom geometry(MultiPolygon,3035)
)""",
    "equipment_data": """CREATE TABLE IF NOT EXISTS public.equipment_data
(
    name varchar(100) PRIMARY KEY,
    s_max_kva integer,
    max_i_a integer,
    r_mohm_per_km integer,
    x_mohm_per_km integer,
    z_mohm_per_km integer,
    cost_eur integer,
    typ varchar(50),
    application_area integer
)""",
    "version": """CREATE TABLE IF NOT EXISTS public.version
(
    version_id varchar(10) PRIMARY KEY,
    version_comment varchar, 
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consumer_categories varchar,
    cable_cost_dict varchar,
    connection_available_cables varchar,   
    other_parameters varchar
)""",
    "classification_version": """CREATE TABLE IF NOT EXISTS public.classification_version
(
    classification_id integer NOT NULL,
    version_comment varchar, 
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    classification_region varchar,
    CONSTRAINT classification_pkey PRIMARY KEY (classification_id)
)""",
    "postcode": """CREATE TABLE IF NOT EXISTS public.postcode
(
    postcode_id integer NOT NULL,
    plz int UNIQUE NOT NULL,
    note varchar,
    qkm double precision,
    population integer,
    geom geometry(MultiPolygon,3035),
    CONSTRAINT "plz-5stellig_pkey" PRIMARY KEY (postcode_id)
)""",
    "postcode_result": """CREATE TABLE IF NOT EXISTS public.postcode_result
(   
    version_id varchar(10) NOT NULL,
    postcode_result_plz integer NOT NULL,
    settlement_type integer,
    geom geometry(MultiPolygon,3035),
    house_distance numeric,
    CONSTRAINT "postcode_result_pkey" PRIMARY KEY (version_id, postcode_result_plz),
    CONSTRAINT fk_postcode_result_version_id
        FOREIGN KEY (version_id)
        REFERENCES public.version (version_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_postcode_result_plz
        FOREIGN KEY (postcode_result_plz)
        REFERENCES public.postcode (plz)
        ON DELETE CASCADE
)""",
    # old name: building_clusters, got merged with grids
    "grid_result": """CREATE TABLE IF NOT EXISTS public.grid_result
(
    grid_result_id SERIAL PRIMARY KEY,
    version_id varchar(10) NOT NULL,
    kcid integer NOT NULL,
    bcid integer NOT NULL,
    plz integer NOT NULL,
    transformer_rated_power bigint,
    model_status integer,
    ont_vertice_id bigint,
    grid json,
    CONSTRAINT cluster_identifier UNIQUE (version_id, kcid, bcid, plz),
    CONSTRAINT unique_grid_result_id_version_id UNIQUE (version_id, grid_result_id),
    CONSTRAINT fk_grid_result_version_id_plz
        FOREIGN KEY (version_id, plz)
        REFERENCES public.postcode_result (version_id, postcode_result_plz)
        ON DELETE CASCADE
);
CREATE INDEX idx_grid_result_version_id_plz_bcid_kcid
ON public.grid_result (version_id, plz, bcid, kcid)
""",
    "lines_result": """CREATE TABLE IF NOT EXISTS public.lines_result
(
    grid_result_id bigint NOT NULL,
    geom geometry(LineString,3035),
    line_name varchar(15),
    std_type varchar(15),
    from_bus integer,
    to_bus integer,
    length_km numeric,
    CONSTRAINT fk_lines_result_grid_result
        FOREIGN KEY (grid_result_id)
        REFERENCES public.grid_result (grid_result_id)
        ON DELETE CASCADE
)""",
    "consumer_categories": """CREATE TABLE IF NOT EXISTS public.consumer_categories
(
    consumer_category_id integer PRIMARY KEY,
    definition varchar(30) UNIQUE NOT NULL,
    peak_load numeric(10,2),
    yearly_consumption numeric(10,2),
    peak_load_per_m2 numeric(10,2),
    yearly_consumption_per_m2 numeric(10,2),
    sim_factor numeric(10,2) NOT NULL
)""",
    "buildings_result": """CREATE TABLE IF NOT EXISTS public.buildings_result
(
    version_id varchar(10) NOT NULL,
    osm_id varchar NOT NULL,
    grid_result_id bigint NOT NULL,
    area numeric,
    type varchar(30),
    geom geometry(MultiPolygon,3035),
    houses_per_building integer,
    center geometry(Point,3035),
    peak_load_in_kw numeric,
    vertice_id integer,
    floors integer,
    connection_point integer,
    CONSTRAINT buildings_result_pkey PRIMARY KEY (version_id, osm_id),
    CONSTRAINT fk_buildings_result_grid_result
        FOREIGN KEY (version_id, grid_result_id)
        REFERENCES public.grid_result (version_id, grid_result_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_buildings_result_type
        FOREIGN KEY (type)
        REFERENCES public.consumer_categories (definition)
        ON DELETE CASCADE
);
CREATE INDEX idx_buildings_result_grid_result_id
ON public.buildings_result (grid_result_id);
""",
    "municipal_register": """CREATE TABLE IF NOT EXISTS public.municipal_register     
(
    plz integer,
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
)""",
    "sample_set": """CREATE TABLE IF NOT EXISTS public.sample_set
(
    classification_id integer NOT NULL,
    plz integer NOT NULL,
    ags bigint,
    bin_no int,
    bins numeric,
    perc_bin numeric,
    count numeric,
    perc numeric,
    CONSTRAINT sample_set_pkey PRIMARY KEY (classification_id, plz),
    CONSTRAINT fk_sample_set_classification_id
        FOREIGN KEY (classification_id)
        REFERENCES public.classification_version (classification_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_sample_set_plz
        FOREIGN KEY (plz, ags)
        REFERENCES public.municipal_register (plz, ags)
        ON DELETE CASCADE
)""",
    "clustering_parameters": """CREATE TABLE IF NOT EXISTS public.clustering_parameters
(
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
        REFERENCES public.grid_result (grid_result_id)
        ON DELETE CASCADE
)""",
    "transformers": """CREATE TABLE IF NOT EXISTS public.transformers
(
    osm_id varchar PRIMARY KEY,
    area double precision,
    power varchar,
    geom_type varchar,
    within_shopping boolean,
    geom geometry(MultiPoint, 3035)
)""",
    "transformer_positions": """CREATE TABLE IF NOT EXISTS public.transformer_positions 
(
    grid_result_id bigint PRIMARY KEY,
    geom geometry(Point,3035),
    osm_id varchar,
    "comment" varchar,
    CONSTRAINT fk_tp_grid_result_id
        FOREIGN KEY (grid_result_id)
        REFERENCES public.grid_result (grid_result_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_tp_osm_id
        FOREIGN KEY (osm_id)
        REFERENCES public.transformers (osm_id)
)""",
    "transformer_classified": """CREATE TABLE IF NOT EXISTS public.transformer_classified 
(
    grid_result_id bigint NOT NULL,
    geom geometry(Point,3035),
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
        REFERENCES public.classification_version (classification_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_transformer_classified_grid_result
        FOREIGN KEY (grid_result_id)
        REFERENCES public.grid_result (grid_result_id)
        ON DELETE CASCADE
)""",
    "ags_log": """CREATE TABLE IF NOT EXISTS public.ags_log
(
    ags bigint PRIMARY KEY
)""",
    "ways": """CREATE TABLE IF NOT EXISTS public.ways
(
    clazz integer,
    source integer,
    target integer,
    cost double precision,
    reverse_cost double precision,
    geom geometry(LineString,3035),
    way_id integer PRIMARY KEY
)""",
    "ways_result": """CREATE TABLE IF NOT EXISTS public.ways_result
(
    version_id varchar(10) NOT NULL,
    clazz integer,
    source integer,
    target integer,
    cost double precision,
    reverse_cost double precision,
    geom geometry(LineString,3035),
    way_id integer NOT NULL,
    plz integer,
    CONSTRAINT pk_ways_result PRIMARY KEY (version_id, way_id, plz),
    CONSTRAINT fk_ways_result_version_id_plz
        FOREIGN KEY (version_id, plz)
        REFERENCES public.postcode_result (version_id, postcode_result_plz)
        ON DELETE CASCADE
)""",
    # old name: grid_parameters
    # saves grid parameters for a whole plz for visualization
    "plz_parameters": """CREATE TABLE IF NOT EXISTS public.plz_parameters
(
    version_id varchar(10) NOT NULL,
    plz integer NOT NULL,
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
        REFERENCES public.postcode_result (version_id, postcode_result_plz)
        ON DELETE CASCADE
)""",
}

TEMP_CREATE_QUERIES = {
    "buildings_tem": """CREATE TABLE IF NOT EXISTS public.buildings_tem
(
    osm_id varchar,
    area numeric,
    type varchar(80),
    geom geometry(Geometry,3035),  -- needs to be geometry as multipoint & multipolygon get inserted here
    houses_per_building integer,
    center geometry(Point,3035),
    peak_load_in_kw numeric,
    plz integer,
    vertice_id bigint,
    bcid integer,
    kcid integer,
    floors integer,
    connection_point integer
)""",
    "ways_tem": """CREATE TABLE IF NOT EXISTS public.ways_tem
(
    clazz integer,
    source integer,
    target integer,
    cost double precision,
    reverse_cost double precision,
    geom geometry(LineString,3035),
    way_id integer,
    plz integer
)""",
}