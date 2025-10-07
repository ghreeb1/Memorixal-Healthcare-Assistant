/**
 * Memory Storytelling: Memory Library Script (storytelling-memories.js)
 * Fetches, displays, and provides interactivity for memories.
 * VERSION 3: Includes modern SVG icons inside dynamically generated cards.
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
const memoriesContainer = document.getElementById('memoriesContainer');
const loadingIndicator = document.getElementById('loadingIndicator');
const statusMessages = document.getElementById('statusMessages');
const audioModal = document.getElementById('audioModal');
const audioPlayer = document.getElementById('audioPlayer');

/**
 * Main initializer function that loads memories on page start
 */
async function initializeMemoriesPage() {
    if (!memoriesContainer) {
        console.error("Memories container not found. This script should be on memories.html.");
        return;
    }
    await loadMemories();
}

/**
 * Fetches all memories from the API and displays them
 */
async function loadMemories() {
    showLoading(true);
    memoriesContainer.innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE_URL}/storytelling/memories/`);
        if (!response.ok) {
            throw new Error(`خطأ في الشبكة: ${response.status}`);
        }
        
        const memories = await response.json();
        displayMemories(memories);
        
    } catch (error) {
        console.error('Failed to load memories:', error);
        memoriesContainer.innerHTML = `
            <div class="error-message">
                <h3>❌ خطأ في تحميل الذكريات</h3>
                <p>تأكد من تشغيل الخادم الخلفي وحاول تحديث الصفحة.</p>
                <button class="btn btn-primary" onclick="location.reload()">إعادة المحاولة</button>
            </div>
        `;
    } finally {
        showLoading(false);
    }
}

/**
 * Renders the fetched memories into the DOM
 * @param {Array} memories Array of memory objects from the API
 */
function displayMemories(memories) {
    if (!memories || memories.length === 0) {
        memoriesContainer.innerHTML = `
            <div class="no-memories">
                <h3>📝 لا توجد ذكريات محفوظة بعد</h3>
                <p>اذهب إلى <a href="/storytelling/add" class="memory-link">صفحة الإضافة</a> لحفظ أول ذكرى!</p>
            </div>
        `;
        return;
    }
    
    const memoriesHtml = memories.slice().reverse().map(createMemoryCard).join('');
    memoriesContainer.innerHTML = memoriesHtml;
}

/**
 * Generates the HTML string for a single memory card, now with SVG icons
 * @param {object} memory A memory object
 * @returns {string} The HTML string for the card
 */
function createMemoryCard(memory) {
    const formattedDate = new Date(memory.created_at).toLocaleDateString('ar-SA', {
        year: 'numeric', month: 'long', day: 'numeric'
    });
    
    const imageHtml = memory.image_url 
        ? `<img src="${API_BASE_URL}${memory.image_url}" alt="${escapeHtml(memory.title)}" onerror="this.style.display='none'">` 
        : `<div class="no-image"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" /></svg></div>`;
    
    const summaryHtml = memory.summary 
        ? `<div class="memory-summary"><strong>النسخة المبسطة:</strong><br>${escapeHtml(memory.summary)}</div>` 
        : '';
    
    return `
        <div class="memory-card" data-memory-id="${memory.id}">
            ${imageHtml}
            <div class="memory-content">
                <h3 class="memory-title">${escapeHtml(memory.title)}</h3>
                <p class="memory-description">${escapeHtml(memory.description)}</p>
                ${summaryHtml}
                <div class="memory-actions">
                    <button class="btn btn-secondary" onclick="generateSummary(${memory.id}, this)">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.898 20.573L16.5 21.75l-.398-1.177a3.375 3.375 0 00-2.456-2.456L12.75 18l1.177-.398a3.375 3.375 0 002.456-2.456L16.5 14.25l.398 1.177a3.375 3.375 0 002.456 2.456L20.25 18l-1.177.398a3.375 3.375 0 00-2.456 2.456z" /></svg>
                        <span>تبسيط النص</span>
                    </button>
                    <button class="btn btn-success" onclick="playMemoryAudio(${memory.id}, this)">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z" /></svg>
                        <span>استمع للذكرى</span>
                    </button>
                </div>
                <div class="memory-date">${formattedDate}</div>
            </div>
        </div>
    `;
}

/**
 * Requests a summary for a memory and updates the card
 * @param {number} memoryId The ID of the memory to summarize
 * @param {HTMLElement} button The button that was clicked
 */
async function generateSummary(memoryId, button) {
    const originalContent = button.innerHTML;
    button.innerHTML = 'جاري التبسيط...';
    button.disabled = true;
    
    try {
        // First, get the memory details to extract the description text
        const memoryResponse = await fetch(`${API_BASE_URL}/storytelling/memories/${memoryId}/`);
        if (!memoryResponse.ok) throw new Error(`Failed to fetch memory: ${memoryResponse.status}`);
        
        const memory = await memoryResponse.json();
        const textToSummarize = memory.description;
        
        if (!textToSummarize || !textToSummarize.trim()) {
            showStatus('لا يوجد نص لتبسيطه في هذه الذكرى.', 'warning');
            button.innerHTML = originalContent;
            button.disabled = false;
            return;
        }
        
        // Now send the summarize request with the required text field
        const response = await fetch(`${API_BASE_URL}/storytelling/memories/${memoryId}/summarize/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: textToSummarize
            })
        });
        
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        
        await response.json();
        showStatus('تم تبسيط النص بنجاح!', 'success');
        
        // A simple way to refresh the view is to reload all memories
        await loadMemories();
        
    } catch (error) {
        console.error('Error generating summary:', error);
        showStatus('حدث خطأ أثناء تبسيط النص.', 'error');
        button.innerHTML = originalContent;
        button.disabled = false;
    }
}

/**
 * Requests text-to-speech audio for a memory and plays it in a modal
 * @param {number} memoryId The ID of the memory to play
 * @param {HTMLElement} button The button that was clicked
 */
async function playMemoryAudio(memoryId, button) {
    const originalContent = button.innerHTML;
    button.innerHTML = 'جاري التحميل...';
    button.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/storytelling/memories/${memoryId}/tts/`);
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        
        const result = await response.json();
        
        if (result.audio_url && result.audio_url.endsWith('.mp3')) {
            audioPlayer.src = `${API_BASE_URL}${result.audio_url}`;
            audioPlayer.load();
            audioPlayer.play();
            audioModal.classList.remove('hidden');
        } else {
            showStatus('خدمة الصوت غير متاحة حالياً.', 'warning');
        }
    } catch (error) {
        console.error('Error playing audio:', error);
        showStatus('حدث خطأ أثناء تشغيل الصوت.', 'error');
    } finally {
        button.innerHTML = originalContent;
        button.disabled = false;
    }
}

/**
 * Closes the audio modal and stops the audio
 */
function closeAudioModal() {
    if (audioModal && audioPlayer) {
        audioModal.classList.add('hidden');
        audioPlayer.pause();
        audioPlayer.src = ''; // Clear the source to stop download
    }
}

// Global event listeners for closing the modal
if (audioModal) {
    audioModal.addEventListener('click', (event) => {
        if (event.target === audioModal) closeAudioModal();
    });
}
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !audioModal.classList.contains('hidden')) {
        closeAudioModal();
    }
});


function showLoading(show) {
    if (loadingIndicator) {
      loadingIndicator.style.display = show ? 'flex' : 'none';
    }
}

function showStatus(message, type) {
    if (!statusMessages) return;
    const statusDiv = document.createElement('div');
    statusDiv.className = `status-message status-${type}`;
    statusDiv.textContent = message;
    statusMessages.appendChild(statusDiv);
    
    const timer = setTimeout(() => statusDiv.remove(), 5000);
    statusDiv.addEventListener('click', () => {
        clearTimeout(timer);
        statusDiv.remove();
    });
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Run the initializer when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeMemoriesPage);
