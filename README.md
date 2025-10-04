# Healthcare Dashboard with Real-Time Updates

A comprehensive FastAPI-based healthcare monitoring system with real-time dashboard updates, patient activity tracking, and AI-powered insights.

## 🚀 Features

- **Real-Time Dashboard**: WebSocket-powered live updates when patients interact with the system
- **Patient Activity Tracking**: Automatic tracking of games, chat sessions, storytelling, relaxation, and AI art activities
- **Dual Database Support**: 
  - `users.db` for authentication and activity tracking
  - `memories.db` for storytelling feature
- **AI Integration**: Gemini AI for insights and report generation
- **Multi-Language Support**: English and Arabic interface
- **Comprehensive Testing**: Full pytest suite for real-time functionality

## 📁 Project Structure

```
healthcare-dashboard/
├── main.py                     # Main FastAPI application
├── config.py                   # Configuration settings
├── routes_dashboard.py         # Dashboard API routes
├── storytelling_feature/       # Storytelling module
│   ├── routes.py              # Storytelling routes
│   ├── core/db.py             # Database models and CRUD
│   └── services/ai_services.py # AI services
├── templates/                  # HTML templates
│   ├── unified_dashboard.html  # Main dashboard
│   └── storytelling/          # Storytelling templates
├── static/                     # Static assets
│   ├── js/activity-tracker.js # Client-side activity tracking
│   ├── css/                   # Stylesheets
│   └── images/                # Image uploads
├── tests/                      # Test suite
│   ├── test_realtime_dashboard.py # Real-time functionality tests
│   └── test_chat_and_usage.py     # Existing tests
├── requirements.txt            # Python dependencies
├── demo_realtime.py           # Real-time demo script
└── README.md                  # This file
```

## 🛠️ Installation & Setup

Follow the steps below to set up and run the project on your local machine.

---

### 1. Create & Activate Virtual Environment

```bash
# Create a virtual environment
python -m venv venv
Activate it:

Windows:

bash
Copy code
venv\Scripts\activate
Linux / macOS:

bash
Copy code
source venv/bin/activate
2. Install Dependencies
bash
Copy code
# Upgrade pip to the latest version
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
### 2. Environment Setup (Optional)

Create a `.env` file for optional configurations:

```env
# Optional: Gemini AI API key for enhanced insights
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Hugging Face API key for AI art
HF_API_KEY=your_huggingface_api_key_here

# Security (change in production)
SECRET_KEY=your_secret_key_change_in_production
```

### 3. Database Initialization

The databases will be created automatically on first run:
- `users.db` - User authentication and activity tracking
- `memories.db` - Storytelling memories storage

## 🚀 Running the Application

### Start the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host localhost --port 8000 --reload
```

The application will be available at:
- **Main Application**: http://localhost:8000
- **Dashboard**: http://localhost:8000/dashboard
- **API Documentation**: http://localhost:8000/docs

## 🧪 Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Test real-time dashboard functionality
pytest tests/test_realtime_dashboard.py -v

# Test existing chat and usage functionality
pytest tests/test_chat_and_usage.py -v
```

### Test Real-Time Functionality

```bash
# Run the interactive demo
python demo_realtime.py
```

This demo will:
1. Test WebSocket connections
2. Simulate patient activities
3. Show real-time updates in the dashboard

## 🔄 Real-Time Dashboard Features

### How It Works

1. **WebSocket Connection**: Dashboard connects to `/ws/dashboard` for real-time updates
2. **Activity Tracking**: Patient activities are tracked via JavaScript client (`activity-tracker.js`)
3. **Automatic Broadcasting**: When activities start/end, updates are broadcast to all connected dashboards
4. **Live Metrics**: Dashboard shows live updates of:
   - Today's sessions
   - Weekly statistics
   - Activity scores
   - Health status

### Testing Real-Time Updates

1. **Open Dashboard**: Navigate to http://localhost:8000/dashboard
2. **Open Patient Interface**: In another tab, go to http://localhost:8000/patient
3. **Interact with Activities**: Use games, chat, storytelling features
4. **Watch Dashboard**: See real-time updates appear instantly

### Manual Testing with API

```bash
# Start an activity
curl -X POST http://localhost:8000/api/patient-activity/start \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "1", "activity_type": "games"}'

# End an activity with score
curl -X POST http://localhost:8000/api/patient-activity/end \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "1", "activity_type": "games", "duration": 300, "score": 85}'
```

## 🎯 Key API Endpoints

### Dashboard APIs
- `GET /api/v1/patients/{id}/dashboard` - Get dashboard metrics
- `GET /api/v1/patients/{id}/activities` - Get patient activities
- `GET /api/v1/patients/{id}/ai-insights` - Get AI insights
- `POST /api/report` - Generate comprehensive report

### Real-Time Activity Tracking
- `POST /api/patient-activity/start` - Start activity tracking
- `POST /api/patient-activity/end` - End activity tracking
- `WS /ws/dashboard` - WebSocket for real-time updates

### Feature Usage Tracking (Legacy)
- `POST /api/v1/feature-usage/start` - Start feature usage
- `POST /api/v1/feature-usage/end` - End feature usage

## 🔧 Configuration

### Database Configuration
- **Users Database**: SQLite at `./users.db`
- **Memories Database**: SQLite at `./memories.db`
- Both databases support concurrent access with proper connection pooling

### WebSocket Configuration
- **Endpoint**: `/ws/dashboard`
- **Auto-reconnection**: Up to 5 attempts with 3-second delays
- **Connection Status**: Visual indicator in dashboard

### AI Services
- **Gemini AI**: For insights and report generation (requires API key)
- **Hugging Face**: For AI art generation (optional)
- **Local Fallbacks**: System works without AI services

## 🐛 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   ```
   Solution: Ensure server is running on localhost:8000
   Check firewall settings
   ```

2. **Database Locked Errors**
   ```
   Solution: Ensure no other instances are running
   Check file permissions on .db files
   ```

3. **Real-Time Updates Not Working**
   ```
   Solution: Check browser console for WebSocket errors
   Verify activity-tracker.js is loaded
   Test with demo_realtime.py
   ```

### Debug Mode

Run with debug logging:
```bash
LOG_LEVEL=DEBUG python main.py
```

## 📊 Performance Notes

- **WebSocket Connections**: Supports multiple concurrent dashboard connections
- **Database**: SQLite with WAL mode for better concurrent access
- **Memory Usage**: Optimized for healthcare environments
- **Response Times**: Sub-second real-time updates

## 🔐 Security

- **Authentication**: JWT-based user authentication
- **CORS**: Configured for development (restrict in production)
- **Input Validation**: Pydantic models for API validation
- **SQL Injection**: Protected via SQLAlchemy ORM

## 🚀 Production Deployment

### Environment Variables for Production
```env
DEBUG=False
SECRET_KEY=your_production_secret_key
ALLOWED_ORIGINS=https://yourdomain.com
```

### Recommended Production Setup
- Use PostgreSQL instead of SQLite
- Configure proper CORS origins
- Use HTTPS with SSL certificates
- Set up proper logging and monitoring

## 📝 License

This project is developed for healthcare monitoring purposes. Ensure compliance with healthcare data regulations (HIPAA, GDPR) when deploying in production.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite to identify issues
3. Use the demo script to verify functionality
4. Check server logs for detailed error information
