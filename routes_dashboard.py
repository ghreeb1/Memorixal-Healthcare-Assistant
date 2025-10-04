"""
Enhanced Dashboard API Routes for Healthcare Monitoring System
This module provides all endpoints for patient activity tracking and caregiver monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import os
import json
import traceback
import asyncio
import logging

# Setup logger
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE SETUP
# =============================================================================

DATABASE_URL = "sqlite:///./users.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# =============================================================================
# DATABASE MODELS
# =============================================================================

class Activities(Base):
    __tablename__ = "activities"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    activity_type = Column(String, index=True)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    duration_in_seconds = Column(Integer, nullable=True)
    metadata_json = Column('metadata', Text, nullable=True)

class ActivityMetadata(Base):
    __tablename__ = "activity_metadata"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    icon = Column(String, nullable=True)
    description = Column(String, nullable=True)

# ... (Other models like FeatureUsage can remain as they are)

# ===============================================================================
# HELPER FUNCTIONS
# ===============================================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def setup_initial_activities(target, connection, **kw):
    db_session = SessionLocal()
    try:
        if db_session.query(ActivityMetadata).count() == 0:
            print("Populating initial activity metadata...")
            # ... (Initial activities list)
            initial_activities = [
                ActivityMetadata(id="caregiver", name="Personal Caregiver", icon="fas fa-comments", description="AI-powered caregiver assistance and conversation."),
                ActivityMetadata(id="games", name="Mind Games", icon="fas fa-gamepad", description="Brain training games and cognitive exercises."),
                ActivityMetadata(id="breathing", name="Relaxation Breathing", icon="fas fa-wind", description="Guided breathing exercises for relaxation."),
                ActivityMetadata(id="aiart", name="AI Art Therapy", icon="fas fa-paint-brush", description="Creative art therapy with AI assistance."),
                ActivityMetadata(id="stories", name="Memory Storytelling", icon="fas fa-book-open", description="Interactive storytelling and memory exercises."),
                ActivityMetadata(id="dashboard", name="Caregiver Dashboard", icon="fas fa-chart-pie", description="Caregiver dashboard viewing activity.")
            ]
            db_session.add_all(initial_activities)
            db_session.commit()
    finally:
        db_session.close()

def normalize_activity_id(activity_id: str) -> str:
    if not activity_id: return activity_id
    aid = str(activity_id).strip().lower()
    mapping = { "caregivers": "caregiver", "story": "stories", "storytelling": "stories", "relaxation": "breathing", "ai_art": "aiart" }
    return mapping.get(aid, aid)

Base.metadata.create_all(bind=engine)
with SessionLocal() as db:
    if db.query(ActivityMetadata).count() == 0:
        setup_initial_activities(None, None)

# ==============================================================================
# ✅ GEMINI INTEGRATION HELPERS (FULL IMPLEMENTATION)
# ==============================================================================
def _gemini_available() -> bool:
    """Check if Gemini API key is available"""
    return bool(os.getenv("GEMINI_API_KEY"))

def _call_gemini(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call Gemini API for AI insights"""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            return None
        
        genai.configure(api_key=api_key)
        
        # Try different model names in order of preference
        model_names = [
            "gemini-2.5-pro",
            "gemini-2.5-flash", 
            "gemini-2.0-flash"
        ]

        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
                
                full_prompt = f"{system_prompt}\n\nUSER DATA:\n{user_prompt}"
                resp = model.generate_content(full_prompt, safety_settings=safety_settings)
                
                if resp.text:
                    logger.info(f"Successfully used Gemini model: {model_name}")
                    return resp.text.strip()
                    
            except Exception as model_error:
                logger.warning(f"Failed to use model {model_name}: {model_error}")
                continue
        
        logger.error("All Gemini models failed")
        return None
        
    except Exception as e:
        logger.error(f"Failed to call Gemini API: {e}")
        return None

def _summarize_rows_for_llm(rows: List['Activities']) -> str:
    lines: List[str] = []
    for r in rows[:200]: # Limit prompt size
        when = r.end_time or r.start_time
        ts = when.isoformat() if when else "N/A"
        try:
            md = json.loads(r.metadata_json or "{}")
            score = md.get("score", "N/A")
        except Exception:
            score = "N/A"
        lines.append(f"Timestamp: {ts}, Type: {r.activity_type}, Duration(s): {r.duration_in_seconds}, Score: {score}")
    return "\n".join(lines)
    
# ===============================================================================
# REAL-TIME WEBSOCKET INTEGRATION
# ===============================================================================

# We'll import the connection manager from main.py when the router is included
connection_manager = None

def set_connection_manager(manager):
    """Set the connection manager instance from main.py"""
    global connection_manager
    connection_manager = manager

async def broadcast_dashboard_update(patient_id: str, update_type: str, data: dict):
    """Broadcast real-time updates to dashboard clients"""
    if connection_manager is None:
        return
    
    message = {
        "type": "dashboard_update",
        "update_type": update_type,  # 'activity_started', 'activity_ended', 'data_changed'
        "patient_id": patient_id,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        await connection_manager.broadcast_to_dashboards(message)
    except Exception as e:
        print(f"Error broadcasting dashboard update: {e}")

# ===============================================================================
# API ROUTER
# ===============================================================================
dashboard_router = APIRouter(prefix="/api", tags=["Dashboard"]) 

# =============================================================================
# FEATURE USAGE TRACKING ENDPOINTS (FULL IMPLEMENTATION)
# =============================================================================
@dashboard_router.post("/v1/feature-usage/start")
async def feature_usage_start(request_data: Dict[str, Any], db: Session = Depends(get_db)):
    try:
        patient_id = str(request_data.get("patientId", "1"))  # Default to "1" if not provided
        activity_type = normalize_activity_id(request_data.get("activityType", "unknown"))  # Default activity type
        
        if not patient_id or patient_id == "None":
            patient_id = "1"
        if not activity_type or activity_type == "unknown":
            return JSONResponse({"status": "error", "message": "activityType is required"}, status_code=400)
        
        new_activity = Activities(patient_id=patient_id, activity_type=activity_type, start_time=datetime.utcnow())
        db.add(new_activity)
        db.commit()
        db.refresh(new_activity)
        
        # Broadcast real-time update to dashboards
        try:
            await broadcast_dashboard_update(
                patient_id=patient_id,
                update_type="activity_started",
                data={
                    "activity_id": new_activity.id,
                    "activity_type": activity_type,
                    "start_time": new_activity.start_time.isoformat()
                }
            )
        except Exception as broadcast_error:
            logger.warning(f"Failed to broadcast activity start: {broadcast_error}")
        
        return JSONResponse({"status": "success", "usage_id": new_activity.id})
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": f"Error starting activity: {str(e)}"}, status_code=500)


@dashboard_router.post("/v1/feature-usage/end")
async def feature_usage_end(request_data: Dict[str, Any], db: Session = Depends(get_db)):
    try:
        usage_id = request_data.get("usageId")
        metadata = request_data.get("metadata", {})
        if not usage_id:
            return JSONResponse({"status": "error", "message": "usageId is required"}, status_code=400)
        
        activity_to_end = db.query(Activities).filter(Activities.id == usage_id).first()
        if not activity_to_end:
            return JSONResponse({"status": "error", "message": f"Activity with ID {usage_id} not found"}, status_code=404)
        if activity_to_end.end_time:
             return JSONResponse({"status": "warning", "message": "Activity already ended."})
        
        end_time = datetime.utcnow()
        duration = (end_time - activity_to_end.start_time).total_seconds()
        activity_to_end.end_time = end_time
        activity_to_end.duration_in_seconds = int(duration)
        if metadata:
            activity_to_end.metadata_json = json.dumps(metadata)
        db.commit()
        
        # Broadcast real-time update to dashboards
        try:
            await broadcast_dashboard_update(
                patient_id=activity_to_end.patient_id,
                update_type="activity_ended",
                data={
                    "activity_id": activity_to_end.id,
                    "activity_type": activity_to_end.activity_type,
                    "duration_seconds": activity_to_end.duration_in_seconds,
                    "end_time": end_time.isoformat(),
                    "metadata": metadata
                }
            )
        except Exception as broadcast_error:
            logger.warning(f"Failed to broadcast activity end: {broadcast_error}")
        
        return JSONResponse({"status": "success"})
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": f"Error ending activity: {str(e)}"}, status_code=500)

# =============================================================================
# ACTIVITIES & DASHBOARD DATA ENDPOINTS (FULL IMPLEMENTATION)
# =============================================================================
@dashboard_router.get("/v1/patients/{patient_id}/activities")
async def get_activities_v1(patient_id: str, db: Session = Depends(get_db)):
    # ... (Full implementation from previous correct answer)
    try:
        all_metadata = db.query(ActivityMetadata).all()
        activities: List[Activities] = db.query(Activities).filter(Activities.patient_id == patient_id).all()
        aggregates: Dict[str, Dict[str, Any]] = {}
        for row in activities:
            act = normalize_activity_id(row.activity_type)
            if act not in aggregates:
                aggregates[act] = {"total_sessions": 0, "total_duration": 0, "last_used": None, "scores": []}
            agg = aggregates[act]
            if row.end_time:
                agg["total_sessions"] += 1; agg["total_duration"] += int((row.duration_in_seconds or 0)//60)
                if (agg["last_used"] is None) or (row.end_time > agg["last_used"]): agg["last_used"] = row.end_time
                try:
                    md = json.loads(row.metadata_json or "{}")
                    if "score" in md and md["score"] is not None: agg["scores"].append(int(md["score"]))
                except Exception: pass
        response_activities = []
        for metadata in all_metadata:
            stats = aggregates.get(metadata.id)
            if stats:
                last_used_str = stats["last_used"].strftime("%Y-%m-%d %H:%M") if stats["last_used"] else "N/A"
                avg_duration = (stats["total_duration"]//stats["total_sessions"]) if stats["total_sessions"] else 0
                avg_score = sum(stats["scores"])/len(stats["scores"]) if stats["scores"] else 0
                total_sessions = stats["total_sessions"]
            else: last_used_str, avg_duration, total_sessions, avg_score = "Not Used Yet", 0, 0, 0
            response_activities.append({"id": metadata.id, "name": metadata.name, "icon": metadata.icon, "description": metadata.description, "last_used": last_used_str, "duration": int(avg_duration), "total_sessions": int(total_sessions), "avg_score": round(avg_score, 1)})
        return JSONResponse(status_code=200, content={"status": "success", "activities": response_activities})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching activities: {str(e)}")

@dashboard_router.get("/v1/patients/{patient_id}/dashboard")
async def get_dashboard_data_v1(patient_id: str, db: Session = Depends(get_db)):
    # ... (Full implementation from previous correct answer)
    try:
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_rows: List[Activities] = (db.query(Activities).filter(Activities.patient_id == patient_id, Activities.end_time.isnot(None), Activities.end_time >= seven_days_ago).order_by(Activities.end_time.desc()).all())
        today_date = datetime.utcnow().date()
        today_rows = [r for r in recent_rows if r.end_time and r.end_time.date() == today_date]
        daily_chats = len([r for r in today_rows if normalize_activity_id(r.activity_type) == "caregiver"])
        games_played = len([r for r in today_rows if normalize_activity_id(r.activity_type) == "games"])
        memory_responses = len([r for r in today_rows if normalize_activity_id(r.activity_type) == "stories"])
        scores = [json.loads(r.metadata_json or "{}").get("score") for r in recent_rows if r.metadata_json]
        scores = [int(s) for s in scores if s is not None]
        weekly_stats = {"total_sessions": len(recent_rows), "avg_score": sum(scores)/len(scores) if scores else 0}
        return JSONResponse({"status": "success", "daily_chats": daily_chats, "games_played": games_played, "memory_responses": memory_responses, "weekly_stats": weekly_stats})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard data: {str(e)}")


# ===============================================================================
# ✅ AI INSIGHTS & REPORTING ENDPOINTS (FULL IMPLEMENTATION RE-ADDED)
# ===============================================================================
@dashboard_router.get("/v1/patients/{patient_id}/ai-insights")
async def get_ai_insights_v1(patient_id: str, db: Session = Depends(get_db)):
    try:
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        rows: List[Activities] = (db.query(Activities).filter(Activities.patient_id == patient_id, Activities.end_time.isnot(None), Activities.end_time >= thirty_days_ago).order_by(Activities.end_time).all())
        
        if not rows:
            return JSONResponse({"status": "success", "riskLevel": "Unknown", "summary": "No recent activity data available for analysis.", "recommendations": ["Encourage patient to engage with activities."]})

        total_sessions = len(rows); scores = [json.loads(r.metadata_json or "{}").get("score") for r in rows if r.metadata_json]; scores = [s for s in scores if s is not None]; avg_score = sum(scores)/len(scores) if scores else 0
        risk_level = "High" if avg_score < 60 and total_sessions < 10 else "Medium" if avg_score < 75 else "Low"
        # Generate enhanced local summary
        activity_types = {}
        for row in rows:
            activity_types[row.activity_type] = activity_types.get(row.activity_type, 0) + 1
        
        most_used_activity = max(activity_types.items(), key=lambda x: x[1]) if activity_types else ("none", 0)
        
        summary = f"""**Patient Activity Summary (Last 30 Days)**

📊 **Overview:**
- Total Sessions: {total_sessions}
- Average Performance: {avg_score:.1f}/100
- Most Used Feature: {most_used_activity[0].title()} ({most_used_activity[1]} sessions)

🎯 **Assessment:**
- Engagement Level: {'High' if total_sessions > 20 else 'Moderate' if total_sessions > 10 else 'Low'}
- Performance Trend: {'Excellent' if avg_score >= 80 else 'Good' if avg_score >= 60 else 'Needs Attention'}

💡 **Activity Breakdown:**
{chr(10).join([f"- {activity.title()}: {count} sessions" for activity, count in activity_types.items()])}
"""
        
        recommendations = [
            "Continue monitoring patient progress",
            "Encourage consistent daily engagement" if total_sessions < 15 else "Maintain current engagement level",
            "Focus on lower-scoring activities" if avg_score < 70 else "Excellent performance across activities"
        ]
        provider = "local"
        
        if _gemini_available():
            system_prompt = (
                "You are a clinical assistant AI providing insights for healthcare professionals. "
                "Analyze patient activity data and provide a concise, professional assessment. "
                "Focus on engagement patterns, performance trends, and any concerns. "
                "Keep your response under 200 words and use markdown formatting for better readability."
            )
            user_prompt = f"Patient activity data:\n{_summarize_rows_for_llm(rows)}"
            ai_summary = _call_gemini(system_prompt, user_prompt)
            if ai_summary:
                summary = ai_summary; provider = "gemini"
        
        return JSONResponse({ "status": "success", "riskLevel": risk_level, "summary": summary, "recommendations": recommendations, "provider": provider, "details": { "total_sessions": total_sessions, "avg_score": round(avg_score, 1) }})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating AI insights: {str(e)}")

@dashboard_router.post("/report")
async def generate_report(db: Session = Depends(get_db)):
    try:
        patient_id_for_report = "1"
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        rows: List[Activities] = (db.query(Activities).filter(Activities.patient_id == patient_id_for_report, Activities.end_time.isnot(None), Activities.end_time >= thirty_days_ago).order_by(Activities.end_time.desc()).all())

        if not rows:
            return JSONResponse({"status": "success", "report": "No activity data for the last 30 days."})

        total_sessions = len(rows); scores = [json.loads(r.metadata_json or "{}").get("score") for r in rows if r.metadata_json]; scores = [s for s in scores if s is not None]; avg_score = sum(scores)/len(scores) if scores else 0
        
        # Generate comprehensive local report
        activity_breakdown = {}
        daily_activity = {}
        
        for row in rows:
            # Activity type breakdown
            activity_breakdown[row.activity_type] = activity_breakdown.get(row.activity_type, 0) + 1
            
            # Daily activity tracking
            day = row.end_time.date() if row.end_time else row.start_time.date()
            daily_activity[day] = daily_activity.get(day, 0) + 1
        
        avg_daily_sessions = sum(daily_activity.values()) / len(daily_activity) if daily_activity else 0
        most_active_day = max(daily_activity.items(), key=lambda x: x[1]) if daily_activity else (None, 0)
        
        report = f"""# 🏥 HEALTHCARE PROGRESS REPORT
## Patient Activity Summary (Last 30 Days)

### 📊 Executive Summary
- **Total Sessions**: {total_sessions}
- **Average Performance**: {avg_score:.1f}/100
- **Average Daily Sessions**: {avg_daily_sessions:.1f}
- **Engagement Level**: {'High' if total_sessions > 20 else 'Moderate' if total_sessions > 10 else 'Low'}
- **Performance Rating**: {'Excellent' if avg_score >= 80 else 'Good' if avg_score >= 60 else 'Needs Attention'}

### 🎯 Activity Breakdown
{chr(10).join([f"- **{activity.title()}**: {count} sessions ({count/total_sessions*100:.1f}%)" for activity, count in activity_breakdown.items()])}

### 📈 Engagement Patterns
- **Most Active Day**: {most_active_day[0]} ({most_active_day[1]} sessions)
- **Active Days**: {len(daily_activity)} out of 30 days
- **Consistency Score**: {len(daily_activity)/30*100:.1f}%

### 💡 Key Findings
- Patient shows {'consistent' if len(daily_activity) > 15 else 'irregular'} engagement patterns
- Performance is {'above average' if avg_score > 75 else 'within normal range' if avg_score > 50 else 'below expectations'}
- {'Excellent progress across all activities' if avg_score > 80 else 'Some activities may need additional support' if avg_score < 60 else 'Steady improvement noted'}

### 📋 Recommendations
1. {'Continue current engagement strategy' if total_sessions > 15 else 'Increase daily activity frequency'}
2. {'Maintain excellent performance levels' if avg_score > 80 else 'Focus on improving lower-scoring activities'}
3. {'Consider advanced challenges' if avg_score > 85 else 'Provide additional support for challenging activities'}

---
*Report generated on {datetime.utcnow().strftime('%Y-%m-%d at %H:%M UTC')}*
"""
        
        report_provider = "local"
        if _gemini_available():
            system_prompt = (
                "You are a professional healthcare report writer. Create a comprehensive, "
                "well-structured report based on the provided patient activity data. "
                "Use markdown formatting with clear sections, bullet points, and professional language. "
                "Include executive summary, key findings, trends, and recommendations."
            )
            user_prompt = f"Raw report data:\n\n{report}"
            ai_report = _call_gemini(system_prompt, user_prompt)
            if ai_report:
                report = ai_report; report_provider = "gemini"
        
        return JSONResponse({"status": "success", "report": report, "provider": report_provider})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

# =============================================================================
# ACTIVITY RESULTS ENDPOINTS (MISSING ENDPOINTS)
# =============================================================================

@dashboard_router.get("/v1/patients/{patient_id}/activities/{activity_type}/results")
async def get_activity_results(patient_id: str, activity_type: str, db: Session = Depends(get_db)):
    """Get results for a specific activity type"""
    try:
        normalized_activity = normalize_activity_id(activity_type)
        
        # Get recent activities for this type
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        activities = db.query(Activities).filter(
            Activities.patient_id == patient_id,
            Activities.activity_type == normalized_activity,
            Activities.end_time.isnot(None),
            Activities.end_time >= thirty_days_ago
        ).order_by(Activities.end_time.desc()).limit(10).all()
        
        results = []
        for activity in activities:
            metadata = {}
            if activity.metadata_json:
                try:
                    metadata = json.loads(activity.metadata_json)
                except:
                    metadata = {}
            
            results.append({
                "id": activity.id,
                "start_time": activity.start_time.isoformat(),
                "end_time": activity.end_time.isoformat() if activity.end_time else None,
                "duration_seconds": activity.duration_in_seconds or 0,
                "score": metadata.get("score"),
                "level": metadata.get("level"),
                "metadata": metadata
            })
        
        return JSONResponse({
            "status": "success",
            "activity_type": activity_type,
            "results": results,
            "total_sessions": len(results),
            "avg_score": sum(r["score"] for r in results if r["score"]) / len([r for r in results if r["score"]]) if any(r["score"] for r in results) else 0
        })
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({
            "status": "error", 
            "message": f"Error fetching {activity_type} results: {str(e)}"
        }, status_code=500)

@dashboard_router.post("/v1/patients/{patient_id}/activities")
async def create_patient_activity(patient_id: str, request_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Create a new patient activity (POST endpoint that was missing)"""
    try:
        activity_type = normalize_activity_id(request_data.get("activity_type", "unknown"))
        duration = request_data.get("duration", 0)
        score = request_data.get("score")
        metadata = request_data.get("metadata", {})
        
        if score is not None:
            metadata["score"] = score
        
        # Create completed activity
        new_activity = Activities(
            patient_id=patient_id,
            activity_type=activity_type,
            start_time=datetime.utcnow() - timedelta(seconds=duration),
            end_time=datetime.utcnow(),
            duration_in_seconds=duration,
            metadata_json=json.dumps(metadata) if metadata else None
        )
        
        db.add(new_activity)
        db.commit()
        db.refresh(new_activity)
        
        # Broadcast real-time update
        try:
            await broadcast_dashboard_update(
                patient_id=patient_id,
                update_type="activity_completed",
                data={
                    "activity_id": new_activity.id,
                    "activity_type": activity_type,
                    "duration_seconds": duration,
                    "score": score,
                    "metadata": metadata
                }
            )
        except Exception as broadcast_error:
            logger.warning(f"Failed to broadcast activity completion: {broadcast_error}")
        
        return JSONResponse({
            "status": "success",
            "activity_id": new_activity.id,
            "message": f"Activity {activity_type} recorded successfully"
        })
        
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        return JSONResponse({
            "status": "error",
            "message": f"Error creating activity: {str(e)}"
        }, status_code=500)

# =============================================================================
# HEALTH CHECK AND DEBUG ENDPOINTS
# =============================================================================

@dashboard_router.get("/health")
async def dashboard_health():
    """Health check for dashboard API"""
    return JSONResponse({
        "status": "ok",
        "service": "dashboard_api",
        "connection_manager": "connected" if connection_manager else "not_connected",
        "timestamp": datetime.utcnow().isoformat()
    })

@dashboard_router.get("/debug/connection-status")
async def debug_connection_status():
    """Debug endpoint to check WebSocket connection status"""
    if connection_manager:
        return JSONResponse({
            "status": "ok",
            "connection_manager": "available",
            "active_connections": len(connection_manager.dashboard_connections),
            "timestamp": datetime.utcnow().isoformat()
        })
    else:
        return JSONResponse({
            "status": "error",
            "connection_manager": "not_available",
            "active_connections": 0,
            "timestamp": datetime.utcnow().isoformat()
        })