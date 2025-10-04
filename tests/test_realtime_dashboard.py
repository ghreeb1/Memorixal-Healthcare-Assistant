"""
Test suite for real-time dashboard functionality
Tests WebSocket connections, broadcasting, and patient activity tracking
"""

import pytest
import asyncio
import json
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import WebSocket
from unittest.mock import AsyncMock, MagicMock, patch

# Import the main app and connection manager
from main import app, connection_manager, broadcast_patient_activity_update
from routes_dashboard import broadcast_dashboard_update


class TestWebSocketConnection:
    """Test WebSocket connection and message handling"""
    
    def test_websocket_connection_manager_init(self):
        """Test that connection manager initializes correctly"""
        assert connection_manager is not None
        assert hasattr(connection_manager, 'dashboard_connections')
        assert isinstance(connection_manager.dashboard_connections, list)
    
    @pytest.mark.asyncio
    async def test_websocket_connect_dashboard(self):
        """Test WebSocket connection to dashboard"""
        mock_websocket = AsyncMock(spec=WebSocket)
        
        # Test connection
        await connection_manager.connect_dashboard(mock_websocket)
        
        # Verify websocket was accepted and added to connections
        mock_websocket.accept.assert_called_once()
        assert mock_websocket in connection_manager.dashboard_connections
    
    @pytest.mark.asyncio
    async def test_websocket_disconnect_dashboard(self):
        """Test WebSocket disconnection from dashboard"""
        mock_websocket = AsyncMock(spec=WebSocket)
        
        # Add connection first
        await connection_manager.connect_dashboard(mock_websocket)
        assert mock_websocket in connection_manager.dashboard_connections
        
        # Test disconnection
        connection_manager.disconnect_dashboard(mock_websocket)
        assert mock_websocket not in connection_manager.dashboard_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_to_dashboards(self):
        """Test broadcasting messages to all connected dashboards"""
        mock_websocket1 = AsyncMock(spec=WebSocket)
        mock_websocket2 = AsyncMock(spec=WebSocket)
        
        # Connect multiple websockets
        await connection_manager.connect_dashboard(mock_websocket1)
        await connection_manager.connect_dashboard(mock_websocket2)
        
        test_message = {
            "type": "test_message",
            "data": "test_data",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast message
        await connection_manager.broadcast_to_dashboards(test_message)
        
        # Verify both websockets received the message
        mock_websocket1.send_json.assert_called_once_with(test_message)
        mock_websocket2.send_json.assert_called_once_with(test_message)
    
    @pytest.mark.asyncio
    async def test_broadcast_handles_disconnected_websockets(self):
        """Test that broadcasting handles disconnected websockets gracefully"""
        mock_websocket1 = AsyncMock(spec=WebSocket)
        mock_websocket2 = AsyncMock(spec=WebSocket)
        
        # Make one websocket fail
        mock_websocket1.send_json.side_effect = Exception("Connection closed")
        
        # Connect websockets
        await connection_manager.connect_dashboard(mock_websocket1)
        await connection_manager.connect_dashboard(mock_websocket2)
        
        test_message = {"type": "test", "data": "test"}
        
        # Broadcast should handle the exception and remove failed connection
        await connection_manager.broadcast_to_dashboards(test_message)
        
        # Failed websocket should be removed
        assert mock_websocket1 not in connection_manager.dashboard_connections
        # Working websocket should remain
        assert mock_websocket2 in connection_manager.dashboard_connections
        mock_websocket2.send_json.assert_called_once_with(test_message)


class TestPatientActivityTracking:
    """Test patient activity tracking and real-time updates"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_start_patient_activity_endpoint(self):
        """Test the start patient activity endpoint"""
        with patch('main.broadcast_patient_activity_update') as mock_broadcast:
            mock_broadcast.return_value = asyncio.Future()
            mock_broadcast.return_value.set_result(None)
            
            response = self.client.post(
                "/api/patient-activity/start",
                json={
                    "patient_id": "test_patient",
                    "activity_type": "games"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "games" in data["message"]
            assert "test_patient" in data["message"]
    
    def test_end_patient_activity_endpoint(self):
        """Test the end patient activity endpoint"""
        with patch('main.broadcast_patient_activity_update') as mock_broadcast:
            mock_broadcast.return_value = asyncio.Future()
            mock_broadcast.return_value.set_result(None)
            
            response = self.client.post(
                "/api/patient-activity/end",
                json={
                    "patient_id": "test_patient",
                    "activity_type": "games",
                    "duration": 300,
                    "score": 85
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "games" in data["message"]
            assert "test_patient" in data["message"]
    
    @pytest.mark.asyncio
    async def test_broadcast_patient_activity_update(self):
        """Test the broadcast patient activity update function"""
        with patch.object(connection_manager, 'broadcast_to_dashboards') as mock_broadcast:
            mock_broadcast.return_value = asyncio.Future()
            mock_broadcast.return_value.set_result(None)
            
            await broadcast_patient_activity_update(
                patient_id="test_patient",
                activity_type="games",
                action="started",
                metadata={"score": 85}
            )
            
            # Verify broadcast was called with correct message structure
            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args[0][0]
            
            assert call_args["type"] == "dashboard_update"
            assert call_args["update_type"] == "activity_started"
            assert call_args["patient_id"] == "test_patient"
            assert call_args["data"]["activity_type"] == "games"
            assert call_args["data"]["action"] == "started"
            assert call_args["data"]["metadata"]["score"] == 85
    
    def test_patient_activity_endpoints_error_handling(self):
        """Test error handling in patient activity endpoints"""
        # Test with invalid JSON
        response = self.client.post(
            "/api/patient-activity/start",
            data="invalid json"
        )
        assert response.status_code == 422  # FastAPI validation error
        
        # Test with missing data
        response = self.client.post(
            "/api/patient-activity/start",
            json={}
        )
        assert response.status_code == 200  # Should use defaults
        data = response.json()
        assert data["status"] == "success"


class TestDashboardRoutes:
    """Test dashboard API routes"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_dashboard_data_endpoint(self):
        """Test dashboard data retrieval endpoint"""
        response = self.client.get("/api/v1/patients/1/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "daily_chats" in data
        assert "games_played" in data
        assert "memory_responses" in data
        assert "weekly_stats" in data
    
    def test_activities_endpoint(self):
        """Test activities retrieval endpoint"""
        response = self.client.get("/api/v1/patients/1/activities")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "activities" in data
        assert isinstance(data["activities"], list)
    
    def test_ai_insights_endpoint(self):
        """Test AI insights endpoint"""
        response = self.client.get("/api/v1/patients/1/ai-insights")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "riskLevel" in data
        assert "summary" in data
    
    def test_report_generation_endpoint(self):
        """Test report generation endpoint"""
        response = self.client.post("/api/report")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "report" in data
        assert "provider" in data


class TestRealTimeIntegration:
    """Integration tests for real-time dashboard updates"""
    
    def setup_method(self):
        """Setup test client and mock websockets"""
        self.client = TestClient(app)
        self.mock_websockets = []
    
    @pytest.mark.asyncio
    async def test_end_to_end_activity_tracking(self):
        """Test complete flow from activity start to dashboard update"""
        # Setup mock websocket
        mock_websocket = AsyncMock(spec=WebSocket)
        await connection_manager.connect_dashboard(mock_websocket)
        
        # Start an activity
        with patch('main.broadcast_patient_activity_update', wraps=broadcast_patient_activity_update):
            response = self.client.post(
                "/api/patient-activity/start",
                json={
                    "patient_id": "test_patient",
                    "activity_type": "games"
                }
            )
            
            assert response.status_code == 200
            
            # Give some time for async operations
            await asyncio.sleep(0.1)
            
            # Verify websocket received the message
            mock_websocket.send_json.assert_called()
            call_args = mock_websocket.send_json.call_args[0][0]
            assert call_args["type"] == "dashboard_update"
            assert call_args["update_type"] == "activity_started"
            assert call_args["patient_id"] == "test_patient"
    
    @pytest.mark.asyncio
    async def test_multiple_dashboard_connections(self):
        """Test broadcasting to multiple dashboard connections"""
        # Setup multiple mock websockets
        mock_websockets = [AsyncMock(spec=WebSocket) for _ in range(3)]
        
        for ws in mock_websockets:
            await connection_manager.connect_dashboard(ws)
        
        # Trigger an activity update
        with patch('main.broadcast_patient_activity_update', wraps=broadcast_patient_activity_update):
            response = self.client.post(
                "/api/patient-activity/end",
                json={
                    "patient_id": "test_patient",
                    "activity_type": "games",
                    "duration": 300,
                    "score": 95
                }
            )
            
            assert response.status_code == 200
            
            # Give some time for async operations
            await asyncio.sleep(0.1)
            
            # Verify all websockets received the message
            for ws in mock_websockets:
                ws.send_json.assert_called()
                call_args = ws.send_json.call_args[0][0]
                assert call_args["update_type"] == "activity_ended"
                assert call_args["data"]["metadata"]["score"] == 95


class TestDatabaseConnections:
    """Test database connections and data persistence"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_users_db_connection(self):
        """Test that users.db is accessible"""
        # Test user registration (which uses users.db)
        response = self.client.post(
            "/register",
            data={
                "username": f"test_user_{datetime.now().timestamp()}",
                "password": "test_password"
            }
        )
        # Should redirect (303) on success
        assert response.status_code == 303
    
    def test_memories_db_connection(self):
        """Test that memories.db is accessible through storytelling routes"""
        response = self.client.get("/storytelling/memories/")
        assert response.status_code == 200
        
        # Should return a list (even if empty)
        data = response.json()
        assert isinstance(data, list)


# Fixtures for pytest
@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
async def mock_websocket():
    """Mock WebSocket fixture"""
    mock_ws = AsyncMock(spec=WebSocket)
    yield mock_ws


@pytest.fixture
def clean_connection_manager():
    """Clean connection manager for each test"""
    # Clear all connections before test
    connection_manager.dashboard_connections.clear()
    yield connection_manager
    # Clear all connections after test
    connection_manager.dashboard_connections.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
