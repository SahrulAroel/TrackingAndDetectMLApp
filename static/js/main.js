// Initialize on DOM load
document.addEventListener('DOMContentLoaded', function() {
    initSidebar();
    updateTimestamp();
    setInterval(updateTimestamp, 1000);
    
    // Set current date
    const now = new Date();
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    document.getElementById('currentDate').textContent = now.toLocaleDateString('id-ID', options);
});

// Sidebar functionality
function initSidebar() {
    const menuBtn = document.getElementById('menu-btn');
    const drawer = document.getElementById('nav-drawer');
    const overlay = document.getElementById('drawer-overlay');
    
    if (!menuBtn || !drawer || !overlay) return;
    
    menuBtn.addEventListener('click', function() {
        drawer.classList.toggle('open');
        overlay.classList.toggle('hidden');
        document.body.style.overflow = drawer.classList.contains('open') ? 'hidden' : 'auto';
    });
    
    overlay.addEventListener('click', function() {
        drawer.classList.remove('open');
        overlay.classList.add('hidden');
        document.body.style.overflow = 'auto';
    });
}

// Update timestamp
function updateTimestamp() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('id-ID', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    
    const timestampElements = document.querySelectorAll('#timestamp');
    timestampElements.forEach(el => {
        if (el) el.textContent = timeString;
    });
}

// Show notification
function showNotification(message, type = 'info') {
    const container = document.getElementById('notification-container');
    if (!container) return;
    
    const notification = document.createElement('div');
    const bgColor = type === 'success' ? 'bg-green-500' :
                    type === 'error' ? 'bg-red-500' :
                    type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500';
    
    const icon = type === 'success' ? 'check_circle' :
                 type === 'error' ? 'error' :
                 type === 'warning' ? 'warning' : 'info';
    
    notification.className = `${bgColor} text-white p-4 rounded-lg shadow-lg animate-fade-in`;
    notification.innerHTML = `
        <div class="flex items-center gap-3">
            <span class="material-icons-round">${icon}</span>
            <span>${message}</span>
            <button class="ml-auto hover:opacity-80" onclick="this.parentElement.parentElement.remove()">
                <span class="material-icons-round">close</span>
            </button>
        </div>
    `;
    
    container.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// API Functions
async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        showNotification('Gagal menghubungi server', 'error');
        throw error;
    }
}

// Vehicle detection control
async function startVehicleDetection(type = 'mobil') {
    try {
        const response = await fetchAPI('/api/start', {
            method: 'POST'
        });
        
        showNotification(`Deteksi ${type} berhasil dijalankan`, 'success');
        return response;
    } catch (error) {
        showNotification(`Gagal menjalankan deteksi ${type}`, 'error');
    }
}

async function stopVehicleDetection(type = 'mobil') {
    try {
        const response = await fetchAPI('/api/stop', {
            method: 'POST'
        });
        
        showNotification(`Deteksi ${type} dihentikan`, 'info');
        return response;
    } catch (error) {
        showNotification(`Gagal menghentikan deteksi ${type}`, 'error');
    }
}



// Export functions to window
window.showNotification = showNotification;
window.startVehicleDetection = startVehicleDetection;
window.stopVehicleDetection = stopVehicleDetection;