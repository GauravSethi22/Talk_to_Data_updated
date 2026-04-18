"""
Tests for AI Query System
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json


class TestSemanticCache:
    """Tests for Layer 1: Semantic Cache."""

    def test_cache_initialization(self):
        """Test cache can be initialized."""
        from layers import SemanticCache

        with patch('layers.layer1_semantic_cache.Redis'):
            cache = SemanticCache(
                redis_host="localhost",
                redis_port=6379
            )
            assert cache.similarity_threshold == 0.92
            assert cache.ttl_seconds == 3600

    def test_cache_get_returns_none_when_empty(self):
        """Test cache returns None for empty cache."""
        from layers import SemanticCache

        with patch('layers.layer1_semantic_cache.Redis') as mock_redis:
            mock_instance = MagicMock()
            mock_instance.scan_iter.return_value = []
            mock_redis.return_value = mock_instance

            cache = SemanticCache()
            result = cache.get("test query")
            assert result is None

    def test_cache_stats(self):
        """Test cache statistics."""
        from layers import SemanticCache

        with patch('layers.layer1_semantic_cache.Redis') as mock_redis:
            mock_instance = MagicMock()
            mock_instance.scan_iter.return_value = ["cache:abc", "cache:def"]
            mock_redis.return_value = mock_instance

            cache = SemanticCache()
            stats = cache.get_stats()

            assert "total_entries" in stats
            assert stats["total_entries"] == 2


class TestIntentRouter:
    """Tests for Layer 2: Intent Router."""

    def test_router_initialization(self):
        """Test router can be initialized."""
        from layers import IntentRouter

        router = IntentRouter(model="gpt-4o-mini")
        assert router.model == "gpt-4o-mini"
        assert router.temperature == 0.0

    def test_route_return_types(self):
        """Test router returns valid route types."""
        from layers import IntentRouter, RouteType

        valid_routes = ["sql", "rag", "both"]
        for route in valid_routes:
            assert route in [r.value for r in RouteType]


class TestTAGRetrieval:
    """Tests for Layer 3: TAG Retrieval."""

    def test_table_description(self):
        """Test table description creation."""
        from layers import TableDescription

        table = TableDescription(
            table_name="test_table",
            description="Test table description",
            columns=[
                {"name": "id", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR"}
            ]
        )

        assert table.table_name == "test_table"
        assert len(table.columns) == 2

    def test_table_to_document(self):
        """Test table description to document conversion."""
        from layers import TableDescription

        table = TableDescription(
            table_name="customers",
            description="Customer information",
            columns=[
                {"name": "customer_id", "type": "INTEGER"}
            ]
        )

        doc = table.to_document()
        assert "Table: customers" in doc
        assert "customer_id" in doc


class TestMultiAgentSQL:
    """Tests for Layer 4: Multi-Agent SQL Engine."""

    def test_sql_result_dataclass(self):
        """Test SQLResult dataclass."""
        from layers import SQLResult

        result = SQLResult(
            success=True,
            query="SELECT * FROM customers",
            parameterized_query="SELECT * FROM customers",
            params=[],
            plan="Step 1: Select all customers",
            tables_used=["customers"],
            validation_errors=[],
            message="Success"
        )

        assert result.success is True
        assert result.tables_used == ["customers"]

    def test_validator_dangerous_keywords(self):
        """Test validator detects dangerous keywords."""
        from layers.layer4_multi_agent_sql import MultiAgentSQLEngine, DANGEROUS_KEYWORDS

        # Mock the engine (we don't need real API keys just to test validation logic)
        engine = MultiAgentSQLEngine(api_key="dummy_key")

        # These should be detected as dangerous
        dangerous_queries = [
            "DROP TABLE customers",
            "DELETE FROM orders",
            "INSERT INTO users VALUES (1, 'test')",
            "UPDATE products SET price = 0"
        ]

        for query in dangerous_queries:
            # Create a mock LangGraph state
            state = {
                "user_query": "test",
                "schema_context": "",
                "plan": "",
                "sql_query": query,
                "is_valid": True,
                "validation_errors": [],
                "tables_used": [],
                "parameterized_query": "",
                "params": []
            }

            # Run the validation node
            result_state = engine.validator_node(state)

            # Assert that the query was flagged as invalid
            assert result_state["is_valid"] is False
            assert len(result_state["validation_errors"]) > 0

            # Assert that the error message mentions "Dangerous keyword"
            assert any("Dangerous keyword found" in err for err in result_state["validation_errors"])

class TestSecureExecution:
    """Tests for Layer 5: Secure Execution."""

    def test_execution_result(self):
        """Test ExecutionResult dataclass."""
        from layers import ExecutionResult

        result = ExecutionResult(
            success=True,
            rows=[{"id": 1, "name": "Test"}],
            row_count=1,
            columns=["id", "name"],
            execution_time_ms=50.0
        )

        assert result.success is True
        assert result.row_count == 1
        assert len(result.columns) == 2

    def test_role_manager_sql_generation(self):
        """Test role manager generates correct SQL."""
        from layers import DatabaseRoleManager

        sql = DatabaseRoleManager.create_readonly_role_sql("test_role")

        assert len(sql) > 0
        assert "CREATE ROLE test_role" in sql[0]
        assert "GRANT SELECT" in sql[2]


class TestStoryteller:
    """Tests for Layer 6: Storyteller."""

    def test_lineage_trace(self):
        """Test LineageTrace creation."""
        from layers import LineageTrace

        trace = LineageTrace(
            query="Test query",
            route="sql",
            sql_run="SELECT 1",
            tables_used=["test_table"],
            schemas_retrieved=["test_schema"],
            documents_retrieved=[],
            cache_hit=False,
            cache_similarity=None,
            execution_time_ms=100.0,
            timestamp="2024-01-01T00:00:00"
        )

        assert trace.query == "Test query"
        assert trace.route == "sql"
        assert len(trace.tables_used) == 1

    def test_lineage_to_dict(self):
        """Test lineage can be converted to dict."""
        from layers import LineageTrace

        trace = LineageTrace(
            query="Test",
            route="sql",
            sql_run=None,
            tables_used=[],
            schemas_retrieved=[],
            documents_retrieved=[],
            cache_hit=False,
            cache_similarity=None,
            execution_time_ms=0,
            timestamp="2024-01-01T00:00:00"
        )

        d = trace.to_dict()
        assert isinstance(d, dict)
        assert "query" in d
        assert "route" in d

    def test_query_response(self):
        """Test QueryResponse dataclass."""
        from layers import QueryResponse, LineageTrace

        trace = LineageTrace(
            query="Test",
            route="sql",
            sql_run=None,
            tables_used=[],
            schemas_retrieved=[],
            documents_retrieved=[],
            cache_hit=False,
            cache_similarity=None,
            execution_time_ms=0,
            timestamp="2024-01-01T00:00:00"
        )

        response = QueryResponse(
            answer="Test answer",
            lineage=trace,
            raw_results=[{"id": 1}]
        )

        assert response.answer == "Test answer"
        assert len(response.raw_results) == 1


class TestPipelineIntegration:
    """Integration tests for the main pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        from main_pipeline import AIQuerySystem

        with patch('main_pipeline.SemanticCache'):
            with patch('main_pipeline.IntentRouter'):
                with patch('main_pipeline.TAGRetrieval'):
                    with patch('main_pipeline.MultiAgentSQLEngine'):
                        with patch('main_pipeline.SecureExecutionSandbox'):
                            with patch('main_pipeline.Storyteller'):
                                system = AIQuerySystem(load_sample_schemas=False)
                                assert system is not None

    def test_health_check_returns_dict(self):
        """Test health check returns proper structure."""
        from main_pipeline import AIQuerySystem

        with patch('main_pipeline.SemanticCache') as mock_cache:
            mock_cache_instance = MagicMock()
            mock_cache_instance.is_healthy.return_value = True
            mock_cache.return_value = mock_cache_instance

            with patch('main_pipeline.IntentRouter'):
                with patch('main_pipeline.TAGRetrieval'):
                    with patch('main_pipeline.MultiAgentSQLEngine'):
                        with patch('main_pipeline.SecureExecutionSandbox') as mock_exec:
                            mock_exec_instance = MagicMock()
                            mock_exec_instance.test_connection.return_value = True
                            mock_exec.return_value = mock_exec_instance

                            with patch('main_pipeline.Storyteller'):
                                system = AIQuerySystem(load_sample_schemas=False)
                                health = system.health_check()

                                assert isinstance(health, dict)
                                assert "cache" in health
                                assert "executor" in health


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
