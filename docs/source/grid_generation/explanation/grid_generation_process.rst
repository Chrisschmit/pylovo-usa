Grid Generation Process
=======================

The functionalities of the grid generation process are divided into several main components:

**DatabaseClient** - Main database interface for reading and writing grid data

**GridGenerator** - Core class for generating synthetic low-voltage distribution grids

**ElectricalGridBuilder** - Handles electrical network creation with pandapower and AltDSS

The higher level functions of the GridGenerator are explained in more detail since they contain the assumptions and logic
of grid generation. For a visual representation refer to :doc:`overview`.

Step 1: Region Selection and Data Preprocessing
------------------------------------------------

The selected region is defined using US Census FIPS codes (state, county, county subdivision) in the
:code:`config/config_data.yaml` file. The region boundaries are searched in the :code:`postcode` table
and define the geographical area for which network generation takes place.

Buildings located within the region are selected from the database and stored in :code:`buildings_tem`.
The tables ending with :code:`tem` are temporary tables that store data during grid generation.

For each building:

- House distance is calculated and settlement type is derived
- Settlement type determines which transformer types are installed
- Maximum load is assigned based on building type:

  - **Residential buildings**: Load scaled to households using NREL residential typology data
  - **Commercial/Public/Industrial buildings**: Load based on building floor area

- Buildings without load or with load over 100kW are excluded from the low-voltage network

Finally, existing transformers from OpenStreetMap are transferred to :code:`buildings_tem`.

Step 2: Road Network Processing
--------------------------------

The road network (ways) from OpenStreetMap located in the region area are stored in :code:`ways_tem`.

Connection nodes are created at road intersections and overlapping sections.

Buildings are connected to the road network:

- A path section is created perpendicular from existing roads to the building center
- Each building in :code:`buildings_tem` is assigned a connection node from :code:`ways_tem`

Step 3: K-means Clustering
---------------------------

Since the number of buildings in a region can be too large for a single coherent network,
the buildings are divided into subgroups using the K-means clustering algorithm.

Buildings are clustered based on geographic distance. The number of K-means clusters
for a region is typically in the single digits. Each cluster is assigned a unique ID
(kcid, K-means cluster ID).

Step 4: LV Distribution Transformer Positioning
------------------------------------------------

The first phase of transformer positioning handles low-voltage (LV) distribution transformers
for each K-means cluster (kcid). This creates building clusters (bcid) where each cluster
connects to a single LV transformer.

**Unified Infrastructure Placement**: The system uses a unified infrastructure placement engine
that handles greenfield (new transformer placement) scenarios.

For each kcid:

1. **Settlement Type Determination**: Settlement type is determined for the region, which influences
   transformer selection and placement strategy

2. **Infrastructure Clustering**: Buildings are clustered using the infrastructure placement algorithm:

   - Hierarchical clustering groups buildings based on geographic distance and load requirements
   - Each cluster verifies that transformer capacity is sufficient for all connected consumers
   - Coincidence factors are applied based on consumer types (residential, public, commercial)

3. **Transformer Positioning**: LV transformers are positioned for each building cluster:

   - Results are saved to :code:`lv_grid_result` table
   - Transformer positions are recorded in :code:`transformer_positions` table

Step 5: MV Substation Positioning
----------------------------------

The second phase positions medium-voltage (MV) substations that aggregate multiple LV transformers
and high-load buildings (>100kW) that connect directly to the MV level.

For each kcid:

1. **MV-Level Infrastructure Placement**: The same unified infrastructure placement engine
   creates substation clusters (scid) that group:

   - LV distribution transformers from Step 4
   - Buildings with peak load >100kW (assigned grid_level_connection = MV)

2. **Hierarchical Network Creation**: MV substations create a hierarchical grid structure:

   - Each MV substation (scid) connects multiple LV transformers and MV-level buildings
   - LV transformers are updated with parent scid references
   - Grid topology creates MV → LV → Consumer hierarchy

3. **Database Updates**:

   - Results saved to :code:`grid_result` table
   - Parent-child relationships established between MV substations and LV transformers
   - Model status set to completed for downstream cable installation

Step 6: Hierarchical Cable Installation
-----------------------------------------

Cable installation creates the complete electrical network hierarchy using the **ElectricalGridBuilder**
with backend-agnostic architecture (supports both pandapower and AltDSS backends).

The process builds MV-LV hierarchical grids for each substation cluster (kcid, scid):

**1. MV Network Construction** (Medium Voltage Level):

   - External grid connection established at the MV substation
   - MV buses created for the substation and connection points
   - MV cables connect the substation to LV transformer locations

**2. LV Network Construction** (Low Voltage Level):

   For each LV transformer connected to the MV substation:

   - Distribution transformer created
   - Consumer buildings connected via radial network topology
   - Three-phase allocation balances loads across L1, L2, L3 phases

**3. Cable Routing and Sizing**:

   - **Consumer Connections**: Buildings connected to roads via service drops
   - **Network Routing**: Connection nodes linked along streets using road network from :code:`ways_tem`
   - **Minimal Spanning Tree**: Optimizes network configuration for minimum total cable length
   - **Cable Selection**: Appropriate cable types selected based on current requirements and voltage drop

**4. Parallel Processing**:

   - Cable installation runs in parallel using multiprocessing for efficiency
   - Each worker processes batches of clusters independently
   - Progress tracked per cluster (kcid, scid pair)

The electrical networks are validated through power flow analysis to ensure voltage and
current constraints are met.

Step 7: Save and Finalize Results
----------------------------------

The data from temporary tables (ending with :code:`tem`) are transferred to permanent result tables:

- **buildings_tem** → **buildings_result**: Final building data with grid connections
- **ways_tem** → **ways_result**: Road network with cable routing information
- **lv_grid_result**: LV transformer clusters and building assignments
- **grid_result**: MV substation clusters and hierarchical relationships
- **transformer_positions**: Geographic locations of all transformers

Temporary tables are then cleared and the database transaction is committed.
