from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import logging
import asyncio
import aiohttp
import aiofiles
import json
import time
import uuid
from urllib.parse import urlparse
from pathlib import Path
import threading
from playwright.async_api import async_playwright
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create downloads directory
DOWNLOADS_DIR = Path("downloads")
DOWNLOADS_DIR.mkdir(exist_ok=True)

# In-memory storage for download tasks
download_tasks = {}

class DownloadTask:
    def __init__(self, url, filename, movie_title):
        self.id = str(uuid.uuid4())
        self.url = url
        self.filename = filename
        self.movie_title = movie_title
        self.file_path = DOWNLOADS_DIR / filename
        self.total_size = 0
        self.downloaded = 0
        self.status = "pending"  # pending, downloading, paused, completed, error
        self.error = None
        self.start_time = None
        self.download_thread = None
        self.lock = threading.Lock()
    
    def to_dict(self):
        with self.lock:
            return {
                "id": self.id,
                "url": self.url,
                "filename": self.filename,
                "movie_title": self.movie_title,
                "total_size": self.total_size,
                "downloaded": self.downloaded,
                "progress": round(self.downloaded / self.total_size * 100, 2) if self.total_size > 0 else 0,
                "status": self.status,
                "error": self.error,
                "start_time": self.start_time,
                "elapsed": time.time() - self.start_time if self.start_time else 0
            }

async def extract_download_url(movie_title):
    """Extract direct download URL from MovieBox.ng"""
    logger.info(f"Extracting download URL for: {movie_title}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto('https://moviebox.ng', wait_until='networkidle')
            
            # Search for the movie
            await page.wait_for_selector('.pc-search-input', state='visible', timeout=30000)
            await page.fill('.pc-search-input', movie_title)
            await page.press('.pc-search-input', 'Enter')
            
            try:
                await page.wait_for_selector('.pc-card', state='visible', timeout=15000)
            except Exception:
                logger.info("No results found with exact title, trying with simplified search...")
                # Try with a simplified search
                simplified_title = movie_title.split(' - ')[0].split(':')[0].strip()
                
                await page.click('.pc-search-input', click_count=3)
                await page.fill('.pc-search-input', simplified_title)
                await page.press('.pc-search-input', 'Enter')
                
                await page.wait_for_selector('.pc-card', state='visible', timeout=15000)
            
            # Click on the first movie
            movie_cards = await page.query_selector_all('.pc-card')
            if not movie_cards:
                raise Exception("No search results found.")
            
            await movie_cards[0].click()
            await page.wait_for_selector('.flx-ce-ce.pc-download-btn', state='visible', timeout=15000)
            
            # Click download button
            await page.click('.flx-ce-ce.pc-download-btn')
            await page.wait_for_selector('.pc-select-quality', state='visible', timeout=15000)
            
            # Select highest quality
            resolution_options = await page.query_selector_all('.pc-quality-list .pc-itm')
            if not resolution_options:
                raise Exception("No resolution options found.")
            
            # Enable request interception to capture the download URL
            await page.route('**/*.mp4', lambda route: route.continue_())
            
            # Create a promise to wait for the download URL
            download_url_promise = asyncio.create_task(wait_for_download_url(page))
            
            # Click the download option
            await resolution_options[0].click()
            
            # Wait for the download URL to be captured
            download_url = await asyncio.wait_for(download_url_promise, 30.0)
            
            if not download_url:
                raise Exception("Failed to capture download URL")
            
            logger.info(f"Successfully extracted download URL: {download_url}")
            
            # Get filename from URL
            parsed_url = urlparse(download_url)
            filename = os.path.basename(parsed_url.path)
            if not filename.endswith('.mp4'):
                filename = f"{re.sub(r'[^\w\-_]', '_', movie_title)}.mp4"
            
            await browser.close()
            return download_url, filename
            
        except Exception as e:
            logger.error(f"Error extracting download URL: {str(e)}")
            await browser.close()
            raise e

async def wait_for_download_url(page):
    """Wait for and capture the download URL"""
    download_url = None
    
    def handle_request(route):
        nonlocal download_url
        if route.request.url.endswith('.mp4'):
            download_url = route.request.url
        route.continue_()
    
    await page.route('**/*.mp4', handle_request)
    
    # Wait until we have a download URL or timeout
    start_time = time.time()
    while not download_url and time.time() - start_time < 30:
        await asyncio.sleep(0.5)
    
    return download_url

async def download_file(task):
    """Download a file with progress tracking"""
    task.status = "downloading"
    task.start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(task.url, allow_redirects=True) as response:
                if response.status != 200:
                    task.status = "error"
                    task.error = f"HTTP error: {response.status}"
                    return
                
                task.total_size = int(response.headers.get('Content-Length', 0))
                
                # Create the file
                async with aiofiles.open(task.file_path, 'wb') as f:
                    chunk_size = 1024 * 1024  # 1MB chunks
                    downloaded = 0
                    
                    async for chunk in response.content.iter_chunked(chunk_size):
                        # Check if download should be paused
                        while task.status == "paused":
                            await asyncio.sleep(1)
                            # If status changed to something other than paused or downloading, exit
                            if task.status not in ["paused", "downloading"]:
                                return
                        
                        # If status changed to something other than downloading, exit
                        if task.status != "downloading":
                            return
                        
                        await f.write(chunk)
                        downloaded += len(chunk)
                        task.downloaded = downloaded
                
                task.status = "completed"
                logger.info(f"Download completed: {task.filename}")
    
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        task.status = "error"
        task.error = str(e)

def start_download_thread(task):
    """Start a download in a separate thread"""
    async def download_wrapper():
        await download_file(task)
    
    def run_async_download():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(download_wrapper())
        loop.close()
    
    task.download_thread = threading.Thread(target=run_async_download)
    task.download_thread.daemon = True
    task.download_thread.start()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Start a new download
@app.route('/api/downloads/start', methods=['POST'])
def start_download():
    data = request.json
    movie_title = data.get('movieTitle')
    
    if not movie_title:
        return jsonify({"error": "Movie title is required"}), 400
    
    try:
        # Run the extraction in a separate thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        download_url, filename = loop.run_until_complete(extract_download_url(movie_title))
        loop.close()
        
        # Create download task
        task = DownloadTask(download_url, filename, movie_title)
        download_tasks[task.id] = task
        
        # Start download in a separate thread
        start_download_thread(task)
        
        return jsonify({
            "message": "Download started successfully",
            "download_id": task.id,
            "filename": filename
        }), 200
    
    except Exception as e:
        logger.error(f"Error starting download: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Get all downloads
@app.route('/api/downloads', methods=['GET'])
def get_downloads():
    downloads = [task.to_dict() for task in download_tasks.values()]
    return jsonify({"downloads": downloads}), 200

# Get download status
@app.route('/api/downloads/<download_id>', methods=['GET'])
def get_download_status(download_id):
    task = download_tasks.get(download_id)
    if not task:
        return jsonify({"error": "Download not found"}), 404
    
    return jsonify(task.to_dict()), 200

# Pause download
@app.route('/api/downloads/<download_id>/pause', methods=['POST'])
def pause_download(download_id):
    task = download_tasks.get(download_id)
    if not task:
        return jsonify({"error": "Download not found"}), 404
    
    if task.status == "downloading":
        task.status = "paused"
        return jsonify({"message": "Download paused"}), 200
    else:
        return jsonify({"error": f"Cannot pause download in {task.status} state"}), 400

# Resume download
@app.route('/api/downloads/<download_id>/resume', methods=['POST'])
def resume_download(download_id):
    task = download_tasks.get(download_id)
    if not task:
        return jsonify({"error": "Download not found"}), 404
    
    if task.status == "paused":
        task.status = "downloading"
        return jsonify({"message": "Download resumed"}), 200
    else:
        return jsonify({"error": f"Cannot resume download in {task.status} state"}), 400

# Cancel download
@app.route('/api/downloads/<download_id>/cancel', methods=['POST'])
def cancel_download(download_id):
    task = download_tasks.get(download_id)
    if not task:
        return jsonify({"error": "Download not found"}), 404
    
    task.status = "cancelled"
    
    # Delete the partial file if it exists
    if os.path.exists(task.file_path):
        try:
            os.remove(task.file_path)
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
    
    # Remove from tasks
    del download_tasks[download_id]
    
    return jsonify({"message": "Download cancelled"}), 200

# Stream video (supports partial content for seeking)
@app.route('/api/stream/<download_id>', methods=['GET'])
def stream_video(download_id):
    task = download_tasks.get(download_id)
    if not task:
        return jsonify({"error": "Download not found"}), 404
    
    if not os.path.exists(task.file_path):
        return jsonify({"error": "File not found"}), 404
    
    file_size = os.path.getsize(task.file_path)
    
    # Handle range requests for video seeking
    range_header = request.headers.get('Range', None)
    
    if range_header:
        byte_start, byte_end = 0, None
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()
        
        if groups[0]:
            byte_start = int(groups[0])
        if groups[1]:
            byte_end = int(groups[1])
        
        if byte_end is None:
            byte_end = file_size - 1
        
        length = byte_end - byte_start + 1
        
        resp = Response(
            stream_with_context(partial_read(task.file_path, byte_start, byte_end)),
            status=206,
            mimetype='video/mp4',
            content_type='video/mp4',
            direct_passthrough=True
        )
        
        resp.headers.add('Content-Range', f'bytes {byte_start}-{byte_end}/{file_size}')
        resp.headers.add('Accept-Ranges', 'bytes')
        resp.headers.add('Content-Length', str(length))
        return resp
    
    # Full file response
    return send_file(
        task.file_path,
        mimetype='video/mp4',
        as_attachment=False,
        download_name=task.filename
    )

def partial_read(path, start, end):
    """Generator to read a file partially"""
    with open(path, 'rb') as f:
        f.seek(start)
        remaining = end - start + 1
        chunk_size = 1024 * 1024  # 1MB chunks
        
        while remaining:
            chunk_size = min(chunk_size, remaining)
            data = f.read(chunk_size)
            if not data:
                break
            remaining -= len(data)
            yield data

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)