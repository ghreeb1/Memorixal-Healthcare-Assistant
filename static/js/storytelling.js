/**
 * Memory Storytelling: Add Memory Page Script (storytelling.js)
 * Handles form submission, validation, image preview, and status updates.
 */

// Configuration for API endpoint
const DEFAULT_API_PORT = 8000;
const API_BASE_URL = (() => {
    const override = window.MEMORY_API_BASE_URL;
    if (override) return override.replace(/\/$/, '');
    const origin = window.location.origin;
    if (window.location.pathname.startsWith('/storytelling') || origin.includes(':8000')) {
        return origin;
    }
    return `http://localhost:${DEFAULT_API_PORT}`;
})();

// DOM Elements
const memoryForm = document.getElementById('memoryForm');
const loadingIndicator = document.getElementById('loadingIndicator');
const statusMessages = document.getElementById('statusMessages');
const imageInput = document.getElementById('image');
const imagePreview = document.getElementById('imagePreview');

/**
 * Main initializer function
 */
function initializeApp() {
    if (!memoryForm) {
        console.error("Memory form not found. This script should be on add.html.");
        return;
    }
    memoryForm.addEventListener('submit', handleMemorySubmit);
    if (imageInput) {
        imageInput.addEventListener('change', handleImagePreview);
    }
}

/**
 * Handles the memory form submission
 * @param {Event} event The form submission event
 */
async function handleMemorySubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(memoryForm);
    const title = formData.get('title').trim();
    const description = formData.get('description').trim();
    
    if (!title || !description) {
        showStatus('الرجاء ملء حقلي العنوان والوصف.', 'error');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/storytelling/memories/`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            throw new Error(`خطأ من الخادم: ${response.status}`);
        }
        
        await response.json();
        showStatus('تم حفظ الذكرى بنجاح! ✅', 'success');
        memoryForm.reset();
        if (imagePreview) imagePreview.innerHTML = '';
        
    } catch (error) {
        console.error('Error adding memory:', error);
        showStatus('حدث خطأ أثناء محاولة حفظ الذكرى. الرجاء المحاولة مرة أخرى.', 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Creates a preview of the selected image
 * @param {Event} event The file input change event
 */
function handleImagePreview(event) {
    if (!imagePreview) return;
    const file = event.target.files[0];
    imagePreview.innerHTML = '';
    
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.alt = 'معاينة الصورة';
            imagePreview.appendChild(img);
        };
        reader.readAsDataURL(file);
    }
}

/**
 * Shows or hides the loading indicator
 * @param {boolean} show True to show, false to hide
 */
function showLoading(show) {
    if (loadingIndicator) {
      loadingIndicator.style.display = show ? 'flex' : 'none';
    }
}

/**
 * Displays a status message to the user
 * @param {string} message The message to display
 * @param {'success'|'error'|'info'|'warning'} type The type of the message
 */
function showStatus(message, type) {
    if (!statusMessages) return;
    const statusDiv = document.createElement('div');
    statusDiv.className = `status-message status-${type}`;
    statusDiv.textContent = message;
    
    statusMessages.innerHTML = ''; 
    statusMessages.appendChild(statusDiv);
    
    const timer = setTimeout(() => statusDiv.remove(), 5000);
    statusDiv.addEventListener('click', () => {
        clearTimeout(timer);
        statusDiv.remove();
    });
}

// Run the initializer when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeApp);
