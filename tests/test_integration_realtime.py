"""
Integration Tests for Real-Time Dashboard Functionality
End-to-end testing of the complete real-time pipeline
"""

import pytest
import asyncio
import json
import websockets
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import threading
import time

from main import app, connection_manager

class TestRealTimeIntegration:
    """Integration tests for real-time dashboard functionality"""
    
    def setup_method(self):
        """Setup test client and clean connection manager"""
        self.client = TestClient(app)
        connection_manager.dashboard_connections.clear()
    
    def teardown_method(self):
        """Cleanup after each test"""
        connection_manager.dashboard_connections.clear()
    
    @pytest.mark.asyncio
    async def test_complete_realtime_flow(self):
        """Test the complete real-time flow from activity to dashboard update"""
        
        # Step 1: Setup mock WebSocket connection
        mock_websocket = AsyncMock()
        await connection_manager.connect_dashboard(mock_websocket)
        
        # Step 2: Start a patient activity
        start_response = self.client.post(
            "/api/patient-activity/start",
            json={
                "patient_id": "test_patient_123",
                "activity_type": "games"
            }
        )
        
        assert start_response.status_code == 200
        start_data = start_response.json()
        assert start_data["status"] == "success"
        
        # Give time for async broadcasting
        await asyncio.sleep(0.1)
        
        # Step 3: Verify WebSocket received start message
        assert mock_websocket.send_json.called
        start_message = mock_websocket.send_json.call_args[0][0]
        
        assert start_message["type"] == "dashboard_update"
        assert start_message["update_type"] == "activity_started"
        assert start_message["patient_id"] == "test_patient_123"
        assert start_message["data"]["activity_type"] == "games"
        
        # Reset mock for next call
        mock_websocket.reset_mock()
        
        # Step 4: End the patient activity with score
        end_response = self.client.post(
            "/api/patient-activity/end",
            json={
                "patient_id": "test_patient_123",
                "activity_type": "games",
                "duration": 120,
                "score": 95
            }
        )
        
        assert end_response.status_code == 200
        end_data = end_response.json()
        assert end_data["status"] == "success"
        
        # Give time for async broadcasting
        await asyncio.sleep(0.1)
        
        # Step 5: Verify WebSocket received end message
        assert mock_websocket.send_json.called
        end_message = mock_websocket.send_json.call_args[0][0]
        
        assert end_message["type"] == "dashboard_update"
        assert end_message["update_type"] == "activity_ended"
        assert end_message["patient_id"] == "test_patient_123"
        assert end_message["data"]["activity_type"] == "games"
        assert end_message["data"]["metadata"]["duration"] == 120
        assert end_message["data"]["metadata"]["score"] == 95
    
    @pytest.mark.asyncio
    async def test_multiple_dashboard_connections(self):
        """Test broadcasting to multiple dashboard connections"""
        
        # Setup multiple mock WebSocket connections
        mock_websockets = []
        for i in range(3):
            mock_ws = AsyncMock()
            mock_websockets.append(mock_ws)
            await connection_manager.connect_dashboard(mock_ws)
        
        # Trigger an activity
        response = self.client.post(
            "/api/patient-activity/start",
            json={
                "patient_id": "multi_test",
                "activity_type": "caregiver"
            }
        )
        
        assert response.status_code == 200
        
        # Give time for async broadcasting
        await asyncio.sleep(0.1)
        
        # Verify all WebSockets received the message
        for mock_ws in mock_websockets:
            assert mock_ws.send_json.called
            message = mock_ws.send_json.call_args[0][0]
            assert message["patient_id"] == "multi_test"
            assert message["data"]["activity_type"] == "caregiver"
    
    @pytest.mark.asyncio
    async def test_websocket_connection_failure_handling(self):
        """Test handling of failed WebSocket connections during broadcasting"""
        
        # Setup one working and one failing WebSocket
        working_ws = AsyncMock()
        failing_ws = AsyncMock()
        failing_ws.send_json.side_effect = Exception("Connection lost")
        
        await connection_manager.connect_dashboard(working_ws)
        await connection_manager.connect_dashboard(failing_ws)
        
        # Verify both are connected
        assert len(connection_manager.dashboard_connections) == 2
        
        # Trigger an activity
        response = self.client.post(
            "/api/patient-activity/end",
            json={
                "patient_id": "failure_test",
                "activity_type": "stories",
                "duration": 60,
                "score": 88
            }
        )
        
        assert response.status_code == 200
        
        # Give time for async broadcasting and cleanup
        await asyncio.sleep(0.1)
        
        # Verify working WebSocket received message
        assert working_ws.send_json.called
        
        # Verify failing WebSocket was removed from connections
        assert failing_ws not in connection_manager.dashboard_connections
        assert working_ws in connection_manager.dashboard_connections
        assert len(connection_manager.dashboard_connections) == 1
    
    def test_dashboard_api_endpoints_integration(self):
        """Test that dashboard API endpoints work correctly"""
        
        # Test dashboard data endpoint
        response = self.client.get("/api/v1/patients/1/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "daily_chats" in data
        assert "games_played" in data
        assert "memory_responses" in data
        assert "weekly_stats" in data
        
        # Test activities endpoint
        response = self.client.get("/api/v1/patients/1/activities")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "activities" in data
        assert isinstance(data["activities"], list)
        
        # Test AI insights endpoint
        response = self.client.get("/api/v1/patients/1/ai-insights")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "riskLevel" in data
        assert "summary" in data
    
    def test_feature_usage_endpoints_integration(self):
        """Test the legacy feature usage tracking endpoints"""
        
        # Start feature usage
        start_response = self.client.post(
            "/api/v1/feature-usage/start",
            json={
                "patientId": "legacy_test",
                "activityType": "games"
            }
        )
        
        assert start_response.status_code == 200
        start_data = start_response.json()
        assert start_data["status"] == "success"
        usage_id = start_data["usage_id"]
        
        # End feature usage
        end_response = self.client.post(
            "/api/v1/feature-usage/end",
            json={
                "usageId": usage_id,
                "metadata": {"score": 92, "level": 5}
            }
        )
        
        assert end_response.status_code == 200
        end_data = end_response.json()
        assert end_data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_activity_types_mapping(self):
        """Test that different activity types are handled correctly"""
        
        mock_websocket = AsyncMock()
        await connection_manager.connect_dashboard(mock_websocket)
        
        activity_types = [
            "games",
            "caregiver", 
            "stories",
            "breathing",
            "aiart"
        ]
        
        for activity_type in activity_types:
            # Reset mock for each test
            mock_websocket.reset_mock()
            
            # Test activity
            response = self.client.post(
                "/api/patient-activity/start",
                json={
                    "patient_id": "type_test",
                    "activity_type": activity_type
                }
            )
            
            assert response.status_code == 200
            
            # Give time for async broadcasting
            await asyncio.sleep(0.05)
            
            # Verify correct activity type in message
            assert mock_websocket.send_json.called
            message = mock_websocket.send_json.call_args[0][0]
            assert message["data"]["activity_type"] == activity_type
    
    def test_error_handling_in_endpoints(self):
        """Test error handling in activity endpoints"""
        
        # Test with invalid JSON
        response = self.client.post(
            "/api/patient-activity/start",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # FastAPI validation error
        
        # Test with missing patient_id (should use default)
        response = self.client.post(
            "/api/patient-activity/start",
            json={"activity_type": "games"}
        )
        assert response.status_code == 200  # Should work with defaults
        
        # Test with empty request body
        response = self.client.post(
            "/api/patient-activity/start",
            json={}
        )
        assert response.status_code == 200  # Should work with defaults
    
    @pytest.mark.asyncio
    async def test_websocket_reconnection_scenario(self):
        """Test WebSocket reconnection scenario"""
        
        # Connect a WebSocket
        mock_websocket = AsyncMock()
        await connection_manager.connect_dashboard(mock_websocket)
        
        # Verify connection
        assert len(connection_manager.dashboard_connections) == 1
        
        # Simulate disconnection
        connection_manager.disconnect_dashboard(mock_websocket)
        
        # Verify disconnection
        assert len(connection_manager.dashboard_connections) == 0
        
        # Reconnect
        await connection_manager.connect_dashboard(mock_websocket)
        
        # Verify reconnection
        assert len(connection_manager.dashboard_connections) == 1
        
        # Test that broadcasting still works after reconnection
        response = self.client.post(
            "/api/patient-activity/start",
            json={
                "patient_id": "reconnect_test",
                "activity_type": "games"
            }
        )
        
        assert response.status_code == 200
        
        # Give time for async broadcasting
        await asyncio.sleep(0.1)
        
        # Verify message was received
        assert mock_websocket.send_json.called
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
    
    def test_storytelling_integration(self):
        """Test storytelling feature integration"""
        
        # Test storytelling API
        response = self.client.get("/storytelling/memories/")
        assert response.status_code == 200
        
        # Should return a list (even if empty)
        data = response.json()
        assert isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_concurrent_activities(self):
        """Test handling of concurrent activities from multiple patients"""
        
        mock_websocket = AsyncMock()
        await connection_manager.connect_dashboard(mock_websocket)
        
        # Start multiple activities concurrently
        patients = ["patient_1", "patient_2", "patient_3"]
        activities = ["games", "caregiver", "stories"]
        
        for i, (patient, activity) in enumerate(zip(patients, activities)):
            response = self.client.post(
                "/api/patient-activity/start",
                json={
                    "patient_id": patient,
                    "activity_type": activity
                }
            )
            assert response.status_code == 200
        
        # Give time for all async broadcasts
        await asyncio.sleep(0.2)
        
        # Verify all activities were broadcast
        assert mock_websocket.send_json.call_count == len(patients)


class TestWebSocketEndpoint:
    """Test the WebSocket endpoint directly"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_websocket_endpoint_exists(self):
        """Test that WebSocket endpoint is properly configured"""
        # This test verifies the endpoint exists and is configured
        # Actual WebSocket testing requires a running server
        
        # Test that the WebSocket route is registered
        websocket_routes = [route for route in app.routes if hasattr(route, 'path') and route.path == "/ws/dashboard"]
        assert len(websocket_routes) > 0, "WebSocket route /ws/dashboard not found"


# Fixtures
@pytest.fixture
def clean_connections():
    """Fixture to clean connection manager before and after tests"""
    connection_manager.dashboard_connections.clear()
    yield
    connection_manager.dashboard_connections.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
