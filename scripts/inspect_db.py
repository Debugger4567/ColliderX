"""
Database schema inspector for ColliderX.
Extracts table structures, keys, indexes, and constraints.
"""

from db import get_conn

conn = get_conn()


def get_table_schema(table_name: str):
    """Get complete schema info for a table."""
    conn = get_conn()
    cur = conn.cursor()

    print(f"\n{'='*60}")
    print(f"TABLE: {table_name}")
    print(f"{'='*60}\n")

    # 1. Get column definitions
    cur.execute(f"""
        SELECT 
            column_name,
            data_type,
            character_maximum_length,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position;
    """)

    print("COLUMNS:")
    print("-" * 60)
    columns = cur.fetchall()
    for col in columns:
        name, dtype, max_len, nullable, default = col
        type_str = dtype
        if max_len:
            type_str += f"({max_len})"
        nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
        default_str = f"DEFAULT {default}" if default else ""
        print(f"  {name:20} {type_str:20} {nullable_str:10} {default_str}")

    # 2. Get primary key
    cur.execute(f"""
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = '{table_name}'
            AND tc.constraint_type = 'PRIMARY KEY';
    """)

    pkeys = [row[0] for row in cur.fetchall()]
    if pkeys:
        print(f"\nPRIMARY KEY: {', '.join(pkeys)}")

    # 3. Get indexes
    cur.execute(f"""
        SELECT
            indexname,
            indexdef
        FROM pg_indexes
        WHERE tablename = '{table_name}';
    """)

    indexes = cur.fetchall()
    if indexes:
        print(f"\nINDEXES:")
        print("-" * 60)
        for idx_name, idx_def in indexes:
            print(f"  {idx_name}")
            print(f"    {idx_def}")

    # 4. Get constraints (CHECK, UNIQUE, FOREIGN KEY)
    cur.execute(f"""
        SELECT
            constraint_name,
            constraint_type
        FROM information_schema.table_constraints
        WHERE table_name = '{table_name}'
            AND constraint_type != 'PRIMARY KEY';
    """)

    constraints = cur.fetchall()
    if constraints:
        print(f"\nCONSTRAINTS:")
        print("-" * 60)
        for name, ctype in constraints:
            print(f"  {name:30} ({ctype})")

            # Get constraint definition for CHECK constraints
            if ctype == "CHECK":
                cur.execute(f"""
                    SELECT check_clause
                    FROM information_schema.check_constraints
                    WHERE constraint_name = '{name}';
                """)
                check_def = cur.fetchone()
                if check_def:
                    print(f"    → {check_def[0]}")

    # 5. Get foreign keys
    cur.execute(f"""
        SELECT
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.table_name = '{table_name}'
            AND tc.constraint_type = 'FOREIGN KEY';
    """)

    fkeys = cur.fetchall()
    if fkeys:
        print(f"\nFOREIGN KEYS:")
        print("-" * 60)
        for col, ftable, fcol in fkeys:
            print(f"  {col} → {ftable}({fcol})")

    cur.close()
    conn.close()


def list_all_tables():
    """List all tables in the database."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename;
    """)

    tables = [row[0] for row in cur.fetchall()]

    print("\n" + "=" * 60)
    print("ALL TABLES IN colliderx")
    print("=" * 60)
    for t in tables:
        print(f"  • {t}")

    cur.close()
    conn.close()

    return tables


def generate_create_statements():
    """Generate CREATE TABLE statements for documentation."""
    conn = get_conn()
    cur = conn.cursor()

    # Get all tables
    cur.execute("""
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename;
    """)

    tables = [row[0] for row in cur.fetchall()]

    print("\n" + "=" * 60)
    print("CREATE TABLE STATEMENTS (for documentation)")
    print("=" * 60)

    for table in tables:
        # Get table definition using pg_dump style query
        cur.execute(f"""
            SELECT 
                'CREATE TABLE ' || '{table}' || ' (' ||
                string_agg(
                    column_name || ' ' || 
                    data_type || 
                    CASE WHEN character_maximum_length IS NOT NULL 
                        THEN '(' || character_maximum_length || ')' 
                        ELSE '' END ||
                    CASE WHEN is_nullable = 'NO' THEN ' NOT NULL' ELSE '' END ||
                    CASE WHEN column_default IS NOT NULL 
                        THEN ' DEFAULT ' || column_default 
                        ELSE '' END,
                    ', '
                ) || ');'
            FROM information_schema.columns
            WHERE table_name = '{table}'
            GROUP BY table_name;
        """)

        result = cur.fetchone()
        if result:
            print(f"\n{result[0]}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    # List all tables first
    tables = list_all_tables()

    # Inspect each table
    for table in tables:
        get_table_schema(table)

    # Generate CREATE statements
    generate_create_statements()
