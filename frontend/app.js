const API_BASE = '';

// State
let state = {
    token: localStorage.getItem('token') || null,
    username: localStorage.getItem('username') || null
};

// DOM Elements
const views = {
    login: document.getElementById('login-view'),
    main: document.getElementById('main-view')
};

const tabs = {
    'dashboard-tab': document.getElementById('dashboard-tab'),
    'leads-tab': document.getElementById('leads-tab'),
    'customers-tab': document.getElementById('customers-tab')
};

// Init
document.addEventListener('DOMContentLoaded', () => {
    if (state.token) {
        showMainView();
    } else {
        showLoginView();
    }
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    // Login Form
    document.getElementById('login-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('login-btn');
        const err = document.getElementById('login-error');
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        
        btn.innerHTML = '<i class="ph ph-spinner ph-spin"></i><span>Authenticating...</span>';
        btn.disabled = true;
        err.textContent = '';
        
        try {
            const formData = new URLSearchParams();
            formData.append('username', username);
            formData.append('password', password);
            
            const res = await fetch(`${API_BASE}/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: formData
            });
            
            if (!res.ok) throw new Error('Invalid credentials');
            
            const data = await res.json();
            state.token = data.access_token;
            state.username = username;
            
            localStorage.setItem('token', state.token);
            localStorage.setItem('username', state.username);
            
            showToast('Successfully logged in!', 'success');
            showMainView();
        } catch (error) {
            err.textContent = error.message;
        } finally {
            btn.innerHTML = '<span>Sign In</span><i class="ph-bold ph-arrow-right"></i>';
            btn.disabled = false;
        }
    });
    
    // Register Form Toggle
    document.getElementById('show-register').addEventListener('click', (e) => {
        e.preventDefault();
        document.getElementById('login-form').style.display = 'none';
        document.getElementById('register-form').style.display = 'block';
        document.querySelector('.login-card p').textContent = 'Create a new account';
    });

    document.getElementById('show-login').addEventListener('click', (e) => {
        e.preventDefault();
        document.getElementById('register-form').style.display = 'none';
        document.getElementById('login-form').style.display = 'block';
        document.querySelector('.login-card p').textContent = 'Sign in to access intelligence';
    });

    // Register Form Submit
    document.getElementById('register-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('register-btn');
        const err = document.getElementById('register-error');
        
        const username = document.getElementById('reg-username').value;
        const email = document.getElementById('reg-email').value;
        const password = document.getElementById('reg-password').value;
        
        btn.innerHTML = '<i class="ph ph-spinner ph-spin"></i><span>Creating Account...</span>';
        btn.disabled = true;
        err.textContent = '';
        
        try {
            const res = await fetch(`${API_BASE}/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, email, password })
            });
            
            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.detail || 'Registration failed');
            }
            
            showToast('Account created successfully! Please sign in.', 'success');
            document.getElementById('show-login').click();
            document.getElementById('reg-username').value = '';
            document.getElementById('reg-email').value = '';
            document.getElementById('reg-password').value = '';
        } catch (error) {
            err.textContent = error.message;
        } finally {
            btn.innerHTML = '<span>Sign Up</span><i class="ph-bold ph-arrow-right"></i>';
            btn.disabled = false;
        }
    });
    
    // Logout
    document.getElementById('logout-btn').addEventListener('click', () => {
        state.token = null;
        state.username = null;
        localStorage.removeItem('token');
        localStorage.removeItem('username');
        showLoginView();
    });
    
    // Tab Switching
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const target = item.getAttribute('data-tab');
            switchTab(target);
        });
    });

    // User Profile Click
    const userProfile = document.querySelector('.user-profile');
    if (userProfile) {
        userProfile.addEventListener('click', () => {
            showToast('Profile settings are currently managed by IT administrators.', 'info');
        });
        userProfile.style.cursor = 'pointer';
    }

    // Retrain Button
    document.getElementById('train-btn').addEventListener('click', async () => {
        const btn = document.getElementById('train-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="ph ph-spinner ph-spin"></i><span>Initializing...</span>';
        btn.disabled = true;
        
        try {
            const res = await fetch(`${API_BASE}/train`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${state.token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ notes: "Triggered from UI" })
            });
            
            if (!res.ok) {
                let errMessage = 'Training failed';
                try {
                    const data = await res.json();
                    errMessage = data.detail || errMessage;
                } catch(e) {
                    errMessage = `Server error ${res.status}`;
                }
                throw new Error(errMessage);
            }
            
            const startData = await res.json();
            const jobId = startData.job_id;
            
            // Show modal
            const modal = document.getElementById('modal-training-progress');
            const logsDiv = document.getElementById('training-logs');
            logsDiv.innerHTML = 'Initializing training pipeline...<br>';
            modal.classList.add('active');
            
            // Poll for status
            const pollInterval = setInterval(async () => {
                try {
                    const statusRes = await fetchWithAuth(`/train/status/${jobId}`);
                    
                    if (statusRes.logs && statusRes.logs.length > 0) {
                        logsDiv.innerHTML = statusRes.logs.join('<br>') + '<br>';
                        logsDiv.scrollTop = logsDiv.scrollHeight;
                    }
                    
                    if (statusRes.status === 'completed') {
                        clearInterval(pollInterval);
                        showToast('Models retrained successfully!', 'success');
                        setTimeout(() => {
                            modal.classList.remove('active');
                            location.reload();
                        }, 2000);
                    } else if (statusRes.status === 'failed') {
                        clearInterval(pollInterval);
                        showToast('Training failed. See logs.', 'error');
                        btn.innerHTML = originalText;
                        btn.disabled = false;
                        logsDiv.innerHTML += '<br><button onclick="document.getElementById(\'modal-training-progress\').classList.remove(\'active\')" class="btn secondary-btn" style="margin-top:15px; background:#333; color:white; border:none; padding:8px 16px; border-radius:4px; cursor:pointer;">Close Window</button>';
                    }
                } catch (err) {
                    console.error("Polling error", err);
                }
            }, 1500);
            
        } catch (e) {
            console.error(e);
            showToast(e.message, 'error');
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    });
}

// View Management
function showLoginView() {
    views.main.classList.remove('active');
    views.login.classList.add('active');
}

function showMainView() {
    views.login.classList.remove('active');
    views.main.classList.add('active');
    document.getElementById('current-username').textContent = state.username;
    
    // Load initial data
    loadDashboard();
    loadLeads();
    loadCustomers();
}

function switchTab(targetId) {
    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.querySelector(`.nav-item[data-tab="${targetId}"]`).classList.add('active');
    
    // Update tab content
    Object.values(tabs).forEach(tab => tab.classList.remove('active'));
    tabs[targetId].classList.add('active');
    
    // Update title
    const titles = {
        'dashboard-tab': 'Overview Dashboard',
        'leads-tab': 'Top Prospects',
        'customers-tab': 'Retention Watchlist'
    };
    document.getElementById('page-title').textContent = titles[targetId];
}

// Data Fetching
async function fetchWithAuth(url, options = {}) {
    // Bulletproof cache-busting by appending a unique timestamp
    const separator = url.includes('?') ? '&' : '?';
    const noCacheUrl = `${API_BASE}${url}${separator}_t=${new Date().getTime()}`;
    
    const res = await fetch(noCacheUrl, {
        ...options,
        headers: {
            'Authorization': `Bearer ${state.token}`,
            ...(options.headers || {})
        },
        cache: 'no-store'
    });
    if (res.status === 401) {
        document.getElementById('logout-btn').click();
        throw new Error('Session expired');
    }
    
    if (!res.ok) {
        let errMessage = `Server error ${res.status}`;
        try {
            const errData = await res.json();
            errMessage = errData.detail || errMessage;
        } catch (e) {}
        throw new Error(errMessage);
    }
    
    return res.json();
}

async function loadDashboard() {
    try {
        const data = await fetchWithAuth('/dashboard');
        document.getElementById('stat-total-leads').textContent = data.stats.total_leads.toLocaleString();
        document.getElementById('stat-total-customers').textContent = data.stats.total_customers.toLocaleString();
        // Predictions stat removed
        
        if (data.recent_training && data.recent_training.length > 0) {
            const d = new Date(data.recent_training[0].training_datetime);
            document.getElementById('stat-last-trained').textContent = d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        } else {
            document.getElementById('stat-last-trained').textContent = 'Never';
        }

        // Accuracies
        if (data.stats.latest_lead_accuracy) {
            document.getElementById('info-lead-acc').textContent = (data.stats.latest_lead_accuracy * 100).toFixed(1) + '%';
        }
        if (data.stats.latest_customer_accuracy) {
            document.getElementById('info-cust-acc').textContent = (data.stats.latest_customer_accuracy * 100).toFixed(1) + '%';
        }
    } catch (e) {
        console.error(e);
    }
}

async function loadLeads() {
    try {
        const data = await fetchWithAuth('/leads/top20');
        const tbody = document.getElementById('leads-tbody');
        tbody.innerHTML = '';
        
        let htmlStr = '';
        data.forEach(lead => {
            const scorePercent = (lead.propensity_ratio * 100).toFixed(1);
            let badgeClass = 'warning';
            if (lead.propensity_ratio > 0.7) badgeClass = 'success';
            
            htmlStr += `
                <tr>
                    <td><strong>${lead.name}</strong></td>
                    <td><span class="badge ${badgeClass}">${scorePercent}% Score</span></td>
                    <td><i class="ph-fill ph-lightbulb" style="color:var(--warning); margin-right:8px;"></i>${lead.top_reasons[0]}</td>
                    <td>${lead.contact_number}</td>
                </tr>
            `;
        });
        tbody.innerHTML = htmlStr;
    } catch (e) {
        console.error(e);
    }
}

async function loadCustomers() {
    try {
        const data = await fetchWithAuth('/customers/high-risk');
        const tbody = document.getElementById('customers-tbody');
        tbody.innerHTML = '';
        
        let htmlStr = '';
        data.forEach(cust => {
            const scorePercent = (cust.churn_ratio * 100).toFixed(1);
            let badgeClass = 'warning';
            if (cust.churn_ratio > 0.7) badgeClass = 'danger';
            
            htmlStr += `
                <tr>
                    <td><strong>#${cust.customer_id}</strong><br><small style="color:var(--text-secondary)">${cust.name}</small></td>
                    <td><span class="badge ${badgeClass}">${scorePercent}% Risk</span></td>
                    <td><i class="ph-fill ph-warning-circle" style="color:var(--danger); margin-right:8px;"></i>${cust.top_reasons[0]}</td>
                    <td><span class="badge ${cust.sentiment === 'Negative' ? 'danger' : (cust.sentiment === 'Positive' ? 'success' : 'warning')}">${cust.sentiment}</span></td>
                    <td>${cust.contact_number || 'N/A'}</td>
                </tr>
            `;
        });
        tbody.innerHTML = htmlStr;
    } catch (e) {
        console.error(e);
    }
}

// Toast Helper
function showToast(message, type='success', duration=3000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast';
    
    const toastId = 'toast-' + Math.random().toString(36).substr(2, 9);
    toast.id = toastId;
    
    let icon = '';
    if (type === 'success') {
        icon = '<i class="ph-fill ph-check-circle" style="color:var(--success)"></i>';
    } else if (type === 'error') {
        icon = '<i class="ph-fill ph-warning-circle" style="color:var(--danger)"></i>';
    } else if (type === 'warning') {
        icon = '<i class="ph-fill ph-warning" style="color:var(--warning)"></i>';
    } else if (type === 'loading') {
        icon = '<i class="ph ph-spinner ph-spin" style="color:var(--warning)"></i>';
    }
    
    toast.innerHTML = `${icon} <span>${message}</span>`;
    container.appendChild(toast);
    
    if (duration > 0) {
        setTimeout(() => {
            removeToast(toastId);
        }, duration);
    }
    return toastId;
}

function removeToast(toastId) {
    const toast = document.getElementById(toastId);
    if (toast) {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }
}

// Modal Logic
document.addEventListener('DOMContentLoaded', () => {
    // Lead Modal
    const modalLeads = document.getElementById('modal-all-leads');
    const closeLeads = document.getElementById('close-modal-leads');
    const cardLeads = document.getElementById('card-total-leads');
    
    if (cardLeads) {
        cardLeads.addEventListener('click', async () => {
            modalLeads.classList.add('active');
            await loadAllLeads(1);
        });
    }
    if (closeLeads) closeLeads.addEventListener('click', () => modalLeads.classList.remove('active'));

    const prevLeadsBtn = document.getElementById('prev-leads-btn');
    const nextLeadsBtn = document.getElementById('next-leads-btn');
    if (prevLeadsBtn) prevLeadsBtn.addEventListener('click', async () => await loadAllLeads(currentLeadsPage - 1));
    if (nextLeadsBtn) nextLeadsBtn.addEventListener('click', async () => await loadAllLeads(currentLeadsPage + 1));

    // Customer Modal
    const modalCustomers = document.getElementById('modal-all-customers');
    const closeCustomers = document.getElementById('close-modal-customers');
    const cardCustomers = document.getElementById('card-total-customers');
    
    if (cardCustomers) {
        cardCustomers.addEventListener('click', async () => {
            modalCustomers.classList.add('active');
            await loadAllCustomers(1);
        });
    }
    if (closeCustomers) closeCustomers.addEventListener('click', () => modalCustomers.classList.remove('active'));

    const prevCustomersBtn = document.getElementById('prev-customers-btn');
    const nextCustomersBtn = document.getElementById('next-customers-btn');
    if (prevCustomersBtn) prevCustomersBtn.addEventListener('click', async () => await loadAllCustomers(currentCustomersPage - 1));
    if (nextCustomersBtn) nextCustomersBtn.addEventListener('click', async () => await loadAllCustomers(currentCustomersPage + 1));

    // Training History Modal
    const modalTraining = document.getElementById('modal-training-history');
    const closeTraining = document.getElementById('close-modal-training');
    const cardPredictions = document.getElementById('card-total-predictions');
    const cardLastTrained = document.getElementById('card-last-trained');
    
    const openTrainingModal = async () => {
        modalTraining.classList.add('active');
        await loadTrainingHistory();
    };

    if (cardPredictions) cardPredictions.addEventListener('click', openTrainingModal);
    if (cardLastTrained) cardLastTrained.addEventListener('click', openTrainingModal);
    if (closeTraining) closeTraining.addEventListener('click', () => modalTraining.classList.remove('active'));

    // File Upload Logic
    const uploadLeadsInput = document.getElementById('upload-leads-input');
    const uploadCustomersInput = document.getElementById('upload-customers-input');

    if (uploadLeadsInput) {
        uploadLeadsInput.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                await handleFileUpload(e.target.files[0], '/upload/leads', 'Leads');
                e.target.value = ''; // Reset input
            }
        });
    }

    if (uploadCustomersInput) {
        uploadCustomersInput.addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                await handleFileUpload(e.target.files[0], '/upload/customers', 'Customers');
                e.target.value = ''; // Reset input
            }
        });
    }

    const exportLeadsBtn = document.getElementById('export-leads-btn');
    if (exportLeadsBtn) {
        exportLeadsBtn.addEventListener('click', () => {
            handleFileExport('/leads/export', 'leads_export.csv');
        });
    }

    const exportCustomersBtn = document.getElementById('export-customers-btn');
    if (exportCustomersBtn) {
        exportCustomersBtn.addEventListener('click', () => {
            handleFileExport('/customers/export', 'customers_export.csv');
        });
    }
});

async function handleFileUpload(file, endpoint, type) {
    const toastId = showToast(`Uploading ${type} file... Please wait.`, 'loading', 0);
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        });
        
        removeToast(toastId);
        
        if (!response.ok) {
            let errMessage = `Failed to upload ${type}`;
            try {
                const err = await response.json();
                errMessage = err.detail || errMessage;
            } catch (e) {
                errMessage = `Server error ${response.status}. The file might be too large or invalid.`;
            }
            throw new Error(errMessage);
        }
        
        const result = await response.json();
        showToast(`Successfully uploaded ${result.records_inserted} ${type}!`, 'success', 5000);
        
        // Refresh dashboard to show new numbers
        await loadDashboard();
    } catch (error) {
        removeToast(toastId);
        console.error(error);
        showToast(error.message, 'error', 6000);
    }
}

async function handleFileExport(endpoint, filename) {
    showToast(`Exporting ${filename}...`, 'warning');
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(endpoint, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || `Failed to export ${filename}`);
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        
        showToast(`Successfully exported ${filename}!`, 'success');
    } catch (error) {
        console.error(error);
        showToast(error.message, 'error');
    }
}

async function loadTrainingHistory() {
    try {
        const tbody = document.getElementById('training-history-tbody');
        tbody.innerHTML = '<tr><td colspan="5">Loading...</td></tr>';
        
        const data = await fetchWithAuth('/dashboard/training');
        tbody.innerHTML = '';
        
        // Ensure data exists, default to empty array
        const historyList = data.training_history || [];
        
        historyList.forEach(hist => {
            const date = new Date(hist.training_datetime);
            const dateString = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
            const accuracyPercent = (hist.accuracy * 100).toFixed(1);
            const badgeType = hist.model_type === 'lead' ? 'blue' : 'green';
            const totalRecords = hist.lead_records_used + hist.customer_records_used;
            
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${dateString}</strong></td>
                <td><span style="text-transform: capitalize; color: var(--${badgeType})">${hist.model_type}</span></td>
                <td><span class="badge warning">${hist.model_version}</span></td>
                <td><strong>${accuracyPercent}%</strong></td>
                <td>${totalRecords} records</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error(e);
        showToast('Failed to load training history', 'error');
    }
}

let currentLeadsPage = 1;
async function loadAllLeads(page = 1) {
    try {
        currentLeadsPage = page;
        const tbody = document.getElementById('all-leads-tbody');
        tbody.innerHTML = '<tr><td colspan="4">Loading...</td></tr>';
        
        const response = await fetchWithAuth(`/leads/predicted/all?page=${page}&limit=100`);
        tbody.innerHTML = '';
        
        let htmlStr = '';
        const items = response.data || [];
        
        items.forEach(lead => {
            const scorePercent = (lead.propensity_ratio * 100).toFixed(1);
            let badgeClass = 'warning';
            if (lead.propensity_ratio > 0.7) badgeClass = 'success';
            
            htmlStr += `
                <tr>
                    <td><strong>${lead.name}</strong><br><small style="color:var(--text-secondary)">${lead.email}</small></td>
                    <td><span class="badge ${badgeClass}">${scorePercent}% Score</span></td>
                    <td><i class="ph-fill ph-lightbulb" style="color:var(--warning); margin-right:8px;"></i>${lead.top_reasons[0] || 'Unknown'}</td>
                    <td>${lead.contact_number}</td>
                </tr>
            `;
        });
        tbody.innerHTML = htmlStr;
        
        // Update pagination controls
        const indicator = document.getElementById('page-leads-indicator');
        const prevBtn = document.getElementById('prev-leads-btn');
        const nextBtn = document.getElementById('next-leads-btn');
        
        if (indicator) indicator.textContent = `Page ${response.page} of ${response.total_pages || 1}`;
        if (prevBtn) prevBtn.disabled = response.page <= 1;
        if (nextBtn) nextBtn.disabled = response.page >= (response.total_pages || 1);
        
    } catch (e) {
        console.error(e);
        const tbody = document.getElementById('all-leads-tbody');
        if (tbody) tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--danger)">${e.message}</td></tr>`;
        showToast(e.message || 'Failed to load all leads', 'error');
    }
}

let currentCustomersPage = 1;
async function loadAllCustomers(page = 1) {
    try {
        currentCustomersPage = page;
        const tbody = document.getElementById('all-customers-tbody');
        tbody.innerHTML = '<tr><td colspan="5">Loading...</td></tr>';
        
        const response = await fetchWithAuth(`/customers/predicted/all?page=${page}&limit=100`);
        tbody.innerHTML = '';
        
        let htmlStr = '';
        const items = response.data || [];
        
        items.forEach(cust => {
            const scorePercent = (cust.churn_ratio * 100).toFixed(1);
            let badgeClass = 'warning';
            if (cust.churn_ratio > 0.7) badgeClass = 'danger';
            
            htmlStr += `
                <tr>
                    <td><strong>#${cust.customer_id}</strong><br><small style="color:var(--text-secondary)">${cust.name}<br>${cust.email}</small></td>
                    <td><span class="badge ${badgeClass}">${scorePercent}% Risk</span></td>
                    <td><i class="ph-fill ph-warning-circle" style="color:var(--danger); margin-right:8px;"></i>${cust.top_reasons[0] || 'Unknown'}</td>
                    <td><span class="badge ${cust.sentiment === 'Negative' ? 'danger' : (cust.sentiment === 'Positive' ? 'success' : 'warning')}">${cust.sentiment}</span></td>
                    <td>${cust.contact_number || 'N/A'}</td>
                </tr>
            `;
        });
        tbody.innerHTML = htmlStr;
        
        // Update pagination controls
        const indicator = document.getElementById('page-customers-indicator');
        const prevBtn = document.getElementById('prev-customers-btn');
        const nextBtn = document.getElementById('next-customers-btn');
        
        if (indicator) indicator.textContent = `Page ${response.page} of ${response.total_pages || 1}`;
        if (prevBtn) prevBtn.disabled = response.page <= 1;
        if (nextBtn) nextBtn.disabled = response.page >= (response.total_pages || 1);
        
    } catch (e) {
        console.error(e);
        const tbody = document.getElementById('all-customers-tbody');
        if (tbody) tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--danger)">${e.message}</td></tr>`;
        showToast(e.message || 'Failed to load all customers', 'error');
    }
}
