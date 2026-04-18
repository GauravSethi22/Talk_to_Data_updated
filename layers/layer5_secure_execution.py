"""
Layer 5: Secure Execution Sandbox
Read-only database role with parameterized queries
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import sqlalchemy
from sqlalchemy import create_engine, text
import logging


@dataclass
class ExecutionResult:
    """Result from database execution."""
    success: bool
    rows: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    execution_time_ms: float
    error: Optional[str] = None


class SecureExecutionSandbox:
    """
    Secure database execution using:
    - Read-only database role
    - Parameterized queries
    - Connection pooling
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "yourdatabase",
        db_user: str = "ai_readonly",
        db_password: str = "",
        connection_timeout: int = 30,
        max_result_rows: int = 1000
    ):
        """
        Initialize the secure execution sandbox.

        Args:
            db_host: Database host
            db_port: Database port
            db_name: Database name
            db_user: Read-only user for connections
            db_password: User password
            connection_timeout: Connection timeout in seconds
            max_result_rows: Maximum rows to return
        """
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_user = db_user
        self.connection_timeout = connection_timeout
        self.max_result_rows = max_result_rows

        # Build connection string for read-only user
        self.connection_string = (
            f"postgresql://{db_user}:{db_password}@"
            f"{db_host}:{db_port}/{db_name}"
        )

        # SQLAlchemy engine for easy connection management
        self.engine = create_engine(
            self.connection_string,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args={
                "connect_timeout": connection_timeout,
                "options": "-c statement_timeout=30000"  # 30s timeout
            }
        )

        self.logger = logging.getLogger(__name__)

    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def execute(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> ExecutionResult:
        """
        Execute a SQL query securely.

        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)

        Returns:
            ExecutionResult with rows and metadata
        """
        import time
        start_time = time.time()

        try:
            with self.get_connection() as conn:
                # Set role to read-only
                conn.execute(text("SET ROLE ai_readonly"))

                # Enforce SQL LIMIT to prevent DB OOM
                query_upper = query.strip().upper()
                if query_upper.startswith("SELECT") and "LIMIT" not in query_upper:
                    query = f"{query.rstrip(';')} LIMIT {self.max_result_rows}"

                # Execute query with optional parameters
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))

                # Fetch results
                rows = result.fetchmany(self.max_result_rows)

                # Convert to list of dicts
                columns = list(result.keys())
                rows_dict = [dict(zip(columns, row)) for row in rows]

                execution_time = (time.time() - start_time) * 1000

                return ExecutionResult(
                    success=True,
                    rows=rows_dict,
                    row_count=len(rows_dict),
                    columns=columns,
                    execution_time_ms=execution_time
                )

        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            return ExecutionResult(
                success=False,
                rows=[],
                row_count=0,
                columns=[],
                execution_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def execute_with_transaction(
        self,
        queries: List[Tuple[str, Optional[Tuple]]]
    ) -> List[ExecutionResult]:
        """
        Execute multiple queries in a transaction.

        Args:
            queries: List of (query, params) tuples

        Returns:
            List of ExecutionResults
        """
        results = []

        try:
            with self.get_connection() as conn:
                conn.execute(text("SET ROLE ai_readonly"))

                with conn.begin():
                    for query, params in queries:
                        # Enforce SQL LIMIT to prevent DB OOM
                        query_upper = query.strip().upper()
                        if query_upper.startswith("SELECT") and "LIMIT" not in query_upper:
                            query = f"{query.rstrip(';')} LIMIT {self.max_result_rows}"

                        result = conn.execute(text(query), params or {})
                        rows = result.fetchmany(self.max_result_rows)
                        columns = list(result.keys())

                        results.append(ExecutionResult(
                            success=True,
                            rows=[dict(zip(columns, row)) for row in rows],
                            row_count=len(rows),
                            columns=columns,
                            execution_time_ms=0
                        ))

        except Exception as e:
            self.logger.error(f"Transaction failed: {str(e)}")
            results.append(ExecutionResult(
                success=False,
                rows=[],
                row_count=0,
                columns=[],
                execution_time_ms=0,
                error=str(e)
            ))

        return results

    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table."""
        query = """
        SELECT
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_name = :table_name
        ORDER BY ordinal_position
        """
        result = self.execute(query, {"table_name": table_name})
        return {
            "success": result.success,
            "columns": result.rows if result.success else [],
            "error": result.error
        }

    def close(self):
        """Close all connections."""
        self.engine.dispose()


# SQL Role setup utilities
class DatabaseRoleManager:
    """Utilities for managing database roles."""

    @staticmethod
    def create_readonly_role_sql(role_name: str = "ai_readonly") -> List[str]:
        """
        Generate SQL statements to create a read-only role.

        Args:
            role_name: Name of the read-only role

        Returns:
            List of SQL statements
        """
        return [
            # Create role if not exists
            f"DO $$ BEGIN "
            f"CREATE ROLE {role_name} LOGIN; "
            f"EXCEPTION WHEN duplicate_object THEN NULL; "
            f"END $$;",

            # Grant connect on database
            f"GRANT CONNECT ON DATABASE yourdatabase TO {role_name};",

            # Grant select on all tables
            f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role_name};",

            # Grant select on all sequences (needed for some queries)
            f"GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO {role_name};",

            # Set default privileges for future tables
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            f"GRANT SELECT ON TABLES TO {role_name};",

            # Revoke dangerous privileges
            f"REVOKE INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public FROM {role_name};",
        ]

    @staticmethod
    def get_setup_instructions(role_name: str = "ai_readonly") -> str:
        """Get instructions for setting up the read-only role."""
        return f"""
To set up the read-only role, run the following in your PostgreSQL database:

1. Connect to your database:
   psql -U postgres -d yourdatabase

2. Run these commands:

{chr(10).join(DatabaseRoleManager.create_readonly_role_sql(role_name))}

3. Verify the role can connect:
   psql -U {role_name} -d yourdatabase -c "SELECT 1;"

4. Test that it can only read:
   psql -U {role_name} -d yourdatabase -c "SELECT * FROM customers LIMIT 1;"
   psql -U {role_name} -d yourdatabase -c "DELETE FROM customers;"  -- Should fail
"""


# Factory function
def create_secure_executor(config: Dict[str, Any]) -> SecureExecutionSandbox:
    """Create a SecureExecutionSandbox from configuration."""
    return SecureExecutionSandbox(
        db_host=config.get("db_host", "localhost"),
        db_port=config.get("db_port", 5432),
        db_name=config.get("db_name", "yourdatabase"),
        db_user=config.get("db_user", "ai_readonly"),
        db_password=config.get("db_password", ""),
        connection_timeout=config.get("connection_timeout", 30),
        max_result_rows=config.get("max_result_rows", 1000)
    )


if __name__ == "__main__":
    # Example usage
    print("Secure Execution Sandbox")
    print("=" * 50)

    print("\nSQL to create read-only role:")
    print("-" * 50)
    for sql in DatabaseRoleManager.create_readonly_role_sql():
        print(sql)
        print()

    print("\nSetup Instructions:")
    print("-" * 50)
    print(DatabaseRoleManager.get_setup_instructions())

    # Example connection (will fail without actual DB)
    executor = SecureExecutionSandbox(
        db_host="localhost",
        db_name="testdb",
        db_user="ai_readonly"
    )

    if executor.test_connection():
        print("\nDatabase connection: OK")
    else:
        print("\nDatabase connection: Failed (expected without actual DB)")
