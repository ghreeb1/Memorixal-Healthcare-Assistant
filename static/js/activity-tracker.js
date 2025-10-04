/**
 * Activity Tracker - Client-side script for tracking patient activities
 * and sending real-time updates to the dashboard via WebSocket
 */

class ActivityTracker {
    constructor(patientId = '1') {
        this.patientId = patientId;
        this.apiBase = window.location.origin;
        this.currentActivity = null;
        this.activityStartTime = null;
        this.init();
    }

    init() {
        console.log('Activity Tracker initialized for patient:', this.patientId);
        
        // Auto-track page navigation as activities
        this.trackPageActivity();
        
        // Listen for beforeunload to end current activity
        window.addEventListener('beforeunload', () => {
            if (this.currentActivity) {
                this.endActivity(this.currentActivity);
            }
        });
    }

    trackPageActivity() {
        // Determine activity type based on current page
        const path = window.location.pathname;
        let activityType = 'unknown';
        
        if (path.includes('/patient')) {
            activityType = 'patient_interface';
        } else if (path.includes('/games')) {
            activityType = 'games';
        } else if (path.includes('/relaxation')) {
            activityType = 'breathing';
        } else if (path.includes('/ai_art')) {
            activityType = 'aiart';
        } else if (path.includes('/storytelling')) {
            activityType = 'stories';
        } else if (path.includes('/chatbot') || path.includes('/chat')) {
            activityType = 'caregiver';
        }

        if (activityType !== 'unknown') {
            this.startActivity(activityType);
        }
    }

    async startActivity(activityType) {
        try {
            // End previous activity if exists
            if (this.currentActivity) {
                await this.endActivity(this.currentActivity);
            }

            this.currentActivity = activityType;
            this.activityStartTime = new Date();

            // Send start notification to dashboard
            const response = await fetch(`${this.apiBase}/api/patient-activity/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    patient_id: this.patientId,
                    activity_type: activityType
                })
            });

            if (response.ok) {
                console.log(`Started tracking activity: ${activityType}`);
            }
        } catch (error) {
            console.error('Error starting activity tracking:', error);
        }
    }

    async endActivity(activityType, score = null) {
        try {
            if (!this.activityStartTime) {
                return;
            }

            const duration = Math.round((new Date() - this.activityStartTime) / 1000);

            // Send end notification to dashboard
            const payload = {
                patient_id: this.patientId,
                activity_type: activityType,
                duration: duration
            };

            if (score !== null) {
                payload.score = score;
            }

            const response = await fetch(`${this.apiBase}/api/patient-activity/end`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                console.log(`Ended tracking activity: ${activityType}, duration: ${duration}s`);
            }

            this.currentActivity = null;
            this.activityStartTime = null;
        } catch (error) {
            console.error('Error ending activity tracking:', error);
        }
    }

    // Method to manually track specific activities with scores
    async trackActivityWithScore(activityType, score) {
        await this.startActivity(activityType);
        
        // Simulate some activity time
        setTimeout(async () => {
            await this.endActivity(activityType, score);
        }, 1000);
    }

    // Method to track game completion
    async trackGameCompletion(gameType, score, duration = null) {
        const activityType = 'games';
        
        if (duration) {
            // If duration is provided, use it directly
            const payload = {
                patient_id: this.patientId,
                activity_type: activityType,
                duration: duration,
                score: score
            };

            try {
                await fetch(`${this.apiBase}/api/patient-activity/end`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(payload)
                });
                console.log(`Game completed: ${gameType}, score: ${score}, duration: ${duration}s`);
            } catch (error) {
                console.error('Error tracking game completion:', error);
            }
        } else {
            // Use the current activity tracking
            await this.endActivity(activityType, score);
        }
    }

    // Method to track chat interactions
    async trackChatInteraction() {
        await this.trackActivityWithScore('caregiver', 100);
    }

    // Method to track storytelling interactions
    async trackStorytellingInteraction(score = 100) {
        await this.trackActivityWithScore('stories', score);
    }

    // Method to track relaxation session
    async trackRelaxationSession(duration) {
        const activityType = 'breathing';
        
        const payload = {
            patient_id: this.patientId,
            activity_type: activityType,
            duration: duration,
            score: 100 // Relaxation sessions are always considered successful
        };

        try {
            await fetch(`${this.apiBase}/api/patient-activity/end`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });
            console.log(`Relaxation session completed: ${duration}s`);
        } catch (error) {
            console.error('Error tracking relaxation session:', error);
        }
    }
}

// Global instance - can be used across all pages
window.activityTracker = new ActivityTracker();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ActivityTracker;
}
