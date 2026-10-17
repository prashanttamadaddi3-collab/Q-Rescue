"""
Database module for Q-Rescue application.
Handles SQLite schema creation, connections, transactions, and CRUD operations.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "q_rescue.db")

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return a connection with WAL mode and foreign key constraints enabled."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn

def init_db(db_path: str = DB_PATH) -> None:
    """Initialize database tables if they do not exist."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                full_name TEXT,
                role TEXT DEFAULT 'Responder',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
        """)
        
        # 2. Locations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                location_id TEXT PRIMARY KEY,
                location_name TEXT NOT NULL,
                taluk TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                elevation_m REAL NOT NULL,
                population INTEGER NOT NULL,
                distance_to_rescue_base_km REAL NOT NULL,
                historical_flood_indicator INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 3. Observations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_id TEXT NOT NULL,
                observation_date TEXT NOT NULL,
                rainfall_mm REAL NOT NULL,
                temperature_c REAL NOT NULL,
                humidity_percent REAL NOT NULL,
                river_water_level_m REAL NOT NULL,
                available_rescue_teams INTEGER DEFAULT 0,
                available_boats INTEGER DEFAULT 0,
                shelter_capacity INTEGER DEFAULT 0,
                flood_risk_label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (location_id) REFERENCES locations (location_id) ON DELETE CASCADE
            );
        """)
        
        # 4. Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_id TEXT NOT NULL,
                observation_date TEXT NOT NULL,
                model_type TEXT NOT NULL,
                predicted_risk_level TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                raw_quantum_output TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (location_id) REFERENCES locations (location_id) ON DELETE CASCADE
            );
        """)
        
        # 5. Resource inventory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resource_inventory (
                resource_id TEXT PRIMARY KEY,
                resource_type TEXT NOT NULL,
                total_available_units INTEGER NOT NULL,
                allocated_units INTEGER DEFAULT 0,
                unit_cost_or_weight REAL DEFAULT 1.0,
                depot_location TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 6. Optimization runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                algorithm TEXT NOT NULL,
                total_locations_considered INTEGER NOT NULL,
                total_resources_allocated INTEGER NOT NULL,
                total_demand INTEGER NOT NULL,
                total_shortage INTEGER NOT NULL,
                objective_cost REAL,
                constraint_satisfaction_rate REAL,
                execution_time_seconds REAL,
                notes TEXT
            );
        """)
        
        # 7. Allocation results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS allocation_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                location_id TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                priority_rank INTEGER NOT NULL,
                teams_allocated INTEGER DEFAULT 0,
                boats_allocated INTEGER DEFAULT 0,
                ambulances_allocated INTEGER DEFAULT 0,
                food_allocated INTEGER DEFAULT 0,
                medical_kits_allocated INTEGER DEFAULT 0,
                unmet_demand_flag INTEGER DEFAULT 0,
                notes TEXT,
                FOREIGN KEY (run_id) REFERENCES optimization_runs (run_id) ON DELETE CASCADE,
                FOREIGN KEY (location_id) REFERENCES locations (location_id) ON DELETE CASCADE
            );
        """)
        conn.commit()

def seed_initial_data(db_path: str = DB_PATH) -> None:
    """Seed sample flood data and resource inventory if tables are empty."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    flood_csv = os.path.join(base_dir, "data", "sample_flood_data.csv")
    res_csv = os.path.join(base_dir, "data", "sample_resource_inventory.csv")
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM locations;")
        loc_count = cursor.fetchone()[0]
        
        if loc_count == 0 and os.path.exists(flood_csv):
            df_flood = pd.read_csv(flood_csv)
            # Distinct locations
            loc_df = df_flood[['location_id', 'location_name', 'taluk', 'latitude', 'longitude', 
                               'elevation_m', 'population', 'distance_to_rescue_base_km', 
                               'historical_flood_indicator']].drop_duplicates(subset=['location_id'])
            
            for _, row in loc_df.iterrows():
                cursor.execute("""
                    INSERT OR IGNORE INTO locations 
                    (location_id, location_name, taluk, latitude, longitude, elevation_m, population, distance_to_rescue_base_km, historical_flood_indicator)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    row['location_id'], row['location_name'], row['taluk'], 
                    float(row['latitude']), float(row['longitude']), float(row['elevation_m']),
                    int(row['population']), float(row['distance_to_rescue_base_km']), int(row['historical_flood_indicator'])
                ))
            
            # Observations
            for _, row in df_flood.iterrows():
                cursor.execute("""
                    INSERT INTO observations 
                    (location_id, observation_date, rainfall_mm, temperature_c, humidity_percent, river_water_level_m, 
                     available_rescue_teams, available_boats, shelter_capacity, flood_risk_label)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    row['location_id'], str(row['observation_date']), float(row['rainfall_mm']),
                    float(row['temperature_c']), float(row['humidity_percent']), float(row['river_water_level_m']),
                    int(row.get('available_rescue_teams', 0)), int(row.get('available_boats', 0)),
                    int(row.get('shelter_capacity', 0)), str(row.get('flood_risk_label', 'Low'))
                ))
                
        cursor.execute("SELECT COUNT(*) FROM resource_inventory;")
        res_count = cursor.fetchone()[0]
        if res_count == 0 and os.path.exists(res_csv):
            df_res = pd.read_csv(res_csv)
            for _, row in df_res.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO resource_inventory 
                    (resource_id, resource_type, total_available_units, allocated_units, unit_cost_or_weight, depot_location)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, (
                    str(row['resource_id']), str(row['resource_type']), int(row['total_available_units']),
                    int(row.get('allocated_units', 0)), float(row.get('unit_cost_or_weight', 1.0)),
                    str(row.get('depot_location', 'Belagavi Depot'))
                ))
        conn.commit()

def get_all_locations(db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch all location records as a DataFrame."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM locations ORDER BY taluk, location_name;", conn)

def get_latest_observations(db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch the latest observation for each location joined with location metadata."""
    query = """
        SELECT l.*, o.observation_id, o.observation_date, o.rainfall_mm, o.temperature_c, 
               o.humidity_percent, o.river_water_level_m, o.available_rescue_teams, 
               o.available_boats, o.shelter_capacity, o.flood_risk_label
        FROM locations l
        JOIN observations o ON l.location_id = o.location_id
        WHERE o.observation_date = (
            SELECT MAX(o2.observation_date) 
            FROM observations o2 
            WHERE o2.location_id = l.location_id
        )
        ORDER BY l.taluk, l.location_name;
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn)

def get_all_observations(db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch all observations joined with location metadata."""
    query = """
        SELECT o.*, l.location_name, l.taluk, l.latitude, l.longitude, 
               l.elevation_m, l.population, l.distance_to_rescue_base_km, l.historical_flood_indicator
        FROM observations o
        JOIN locations l ON o.location_id = l.location_id
        ORDER BY o.observation_date DESC, l.location_name ASC;
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn)

def save_prediction(location_id: str, observation_date: str, model_type: str, 
                    predicted_risk_level: str, confidence_score: float, 
                    raw_quantum_output: str = "", db_path: str = DB_PATH) -> int:
    """Insert or update a risk prediction."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (location_id, observation_date, model_type, predicted_risk_level, confidence_score, raw_quantum_output)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (location_id, observation_date, model_type, predicted_risk_level, confidence_score, raw_quantum_output))
        conn.commit()
        return cursor.lastrowid

def get_latest_predictions(model_type: Optional[str] = None, db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch the most recent predictions per location."""
    filter_sql = "AND p.model_type = ?" if model_type else ""
    params = [model_type] if model_type else []
    query = f"""
        SELECT p.*, l.location_name, l.taluk, l.latitude, l.longitude, l.population, l.distance_to_rescue_base_km
        FROM predictions p
        JOIN locations l ON p.location_id = l.location_id
        WHERE p.prediction_id = (
            SELECT MAX(p2.prediction_id) 
            FROM predictions p2 
            WHERE p2.location_id = p.location_id {filter_sql.replace('p.', 'p2.')}
        )
        ORDER BY CASE p.predicted_risk_level 
            WHEN 'Critical' THEN 1 
            WHEN 'High' THEN 2 
            WHEN 'Medium' THEN 3 
            WHEN 'Low' THEN 4 
            ELSE 5 END;
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params if model_type else None)

def get_resource_inventory(db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch current resource inventory."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM resource_inventory ORDER BY resource_id;", conn)

def update_resource_inventory(resource_id: str, total_available_units: int, 
                              allocated_units: int = 0, db_path: str = DB_PATH) -> None:
    """Update resource counts."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE resource_inventory 
            SET total_available_units = ?, allocated_units = ?, last_updated = CURRENT_TIMESTAMP
            WHERE resource_id = ?;
        """, (total_available_units, allocated_units, resource_id))
        conn.commit()

def save_optimization_run(algorithm: str, total_locations: int, total_allocated: int, 
                          total_demand: int, total_shortage: int, objective_cost: float,
                          satisfaction_rate: float, exec_time: float, 
                          allocation_items: List[Dict[str, Any]], notes: str = "",
                          db_path: str = DB_PATH) -> int:
    """Store an optimization run along with detailed location allocations in a transaction."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO optimization_runs 
            (algorithm, total_locations_considered, total_resources_allocated, total_demand, total_shortage,
             objective_cost, constraint_satisfaction_rate, execution_time_seconds, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (algorithm, total_locations, total_allocated, total_demand, total_shortage, 
              objective_cost, satisfaction_rate, exec_time, notes))
        run_id = cursor.lastrowid
        
        for item in allocation_items:
            cursor.execute("""
                INSERT INTO allocation_results 
                (run_id, location_id, risk_level, priority_rank, teams_allocated, boats_allocated, 
                 ambulances_allocated, food_allocated, medical_kits_allocated, unmet_demand_flag, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                run_id, item['location_id'], item['risk_level'], item['priority_rank'],
                item.get('teams_allocated', 0), item.get('boats_allocated', 0),
                item.get('ambulances_allocated', 0), item.get('food_allocated', 0),
                item.get('medical_kits_allocated', 0), item.get('unmet_demand_flag', 0),
                item.get('notes', '')
            ))
        conn.commit()
        return run_id

def get_optimization_history(db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch past optimization runs."""
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM optimization_runs ORDER BY run_timestamp DESC;", conn)

def get_run_allocations(run_id: int, db_path: str = DB_PATH) -> pd.DataFrame:
    """Fetch allocation rows for a specific run."""
    query = """
        SELECT a.*, l.location_name, l.taluk, l.latitude, l.longitude, l.population, l.distance_to_rescue_base_km
        FROM allocation_results a
        JOIN locations l ON a.location_id = l.location_id
        WHERE a.run_id = ?
        ORDER BY a.priority_rank ASC;
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=[run_id])
        # Ensure the database schema and sample data are ready
# whenever a Streamlit page imports this module.
init_db()
seed_initial_data()
