// Access the video, canvas, and audio elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const captureButton = document.getElementById('capture');
const recordButton = document.getElementById('record');
const stopButton = document.getElementById('stop');
const audioPlayback = document.getElementById('audioPlayback');
const photoDiv = document.getElementById('photo');
const cameraSelect = document.getElementById('cameraSelect');

// Variables for audio recording
let mediaRecorder;
let audioChunks = [];
let currentStream;

// Get the available video devices (cameras)
async function getAvailableCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        // Log available devices for debugging
        console.log('Available video devices:', videoDevices);

        // Clear the existing camera options
        cameraSelect.innerHTML = '<option value="">Select Camera</option>';

        // If no video devices are available
        if (videoDevices.length === 0) {
            console.log("No video input devices detected.");
            return;
        }

        // Populate the dropdown with the available cameras
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });

        // Set up an event listener for when the user selects a camera
        cameraSelect.addEventListener('change', (event) => {
            const selectedDeviceId = event.target.value;
            if (selectedDeviceId) {
                startCamera(selectedDeviceId);
            }
        });

        // Automatically start the first camera if available
        if (videoDevices.length > 0) {
            startCamera(videoDevices[0].deviceId);
        }

    } catch (error) {
        console.error('Error getting devices:', error);
    }
}

// Start the camera based on the selected device
async function startCamera(deviceId) {
    // If there's an existing stream, stop it
    if (currentStream) {
        const tracks = currentStream.getTracks();
        tracks.forEach(track => track.stop());
    }

    try {
        // Get the selected camera stream
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: deviceId ? { exact: deviceId } : undefined },
            audio: true
        });

        currentStream = stream;  // Save the current stream to stop it later

        // Set the video source to the selected camera stream
        video.srcObject = stream;

        // Ensure the video element is visible and playing
        video.play();
        video.style.display = 'block';

        // Initialize audio recording
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioUrl;
        };

        console.log(`Camera started with device ID: ${deviceId}`);

    } catch (error) {
        console.error('Error accessing the camera and microphone:', error);
        alert('Failed to access the camera. Please check your permissions.');
    }
}

// Capture image from the video feed
captureButton.addEventListener('click', () => {
    // Draw the current frame from video onto the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Show the canvas
    const imageData = canvas.toDataURL('image/png');

    // Create an image element and append it to the photo div
    const img = document.createElement('img');
    img.src = imageData;
    photoDiv.innerHTML = ''; // Clear previous images
    photoDiv.appendChild(img);
});

// Start recording audio
recordButton.addEventListener('click', () => {
    audioChunks = [];  // Reset the audio chunks array
    mediaRecorder.start();
    console.log('Recording started...');
});

// Stop recording audio
stopButton.addEventListener('click', () => {
    mediaRecorder.stop();
    console.log('Recording stopped.');
});

// Initialize the camera list on page load
getAvailableCameras();
