document.addEventListener('DOMContentLoaded', function () {
    const video = document.getElementById('video');
    const sessionBtn = document.getElementById('sessionBtn');
    const statusText = document.getElementById('status');
    const engagementScoreElement = document.getElementById('engagement-score');
    const averageDurationElement = document.getElementById('average-duration');
    let startTime;
    let mediaRecorder;
    let videoChunks = [];
    let recordingInProgress = false;
    let autoStopTimeout;
    let dataSent = true;
    let blob;
    let earlyStop;

    // Function to start video recording
    function startRecording() {
        if (mediaRecorder && mediaRecorder.state === 'inactive') {
            mediaRecorder.start();
            startTime = new Date();
            dataSent = false;
            statusText.innerText = "Recording in progress...";
            recordingInProgress = true;

            // Automatically stop recording after 6 seconds
            autoStopTimeout = setTimeout(function() {
                if (mediaRecorder.state === 'recording') {
                    earlyStop = true
                    mediaRecorder.stop();
                    // Do not send data, just stop recording
                    // statusText.innerText = "Recording stopped. Click 'Next Image' to process.";
                }
            }, 6000);
        } else {
            console.error('MediaRecorder is not ready or missing:', mediaRecorder?.state);
            // statusText.innerText = "Recorder not ready. Please wait.";
        }
    }

    // Event listener for session button
    sessionBtn.addEventListener('click', function () {
        if (!recordingInProgress && dataSent) {
            dataSent = false
            startRecording();
            sessionBtn.innerText = "Next Image";
        } else {
            clearTimeout(autoStopTimeout); 
            if (recordingInProgress && mediaRecorder.state === 'recording') {
                earlyStop = false
                mediaRecorder.stop();
            }

            if (earlyStop){
                sendData(blob);
            }
        }
    });

    // Initialize video capture
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, frameRate: 60 } })
            .then(function (stream) {
                video.srcObject = stream;
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = (e) => videoChunks.push(e.data);
                mediaRecorder.onstop = handleRecordingStop;
            })
            .catch(function (err) {
                console.error("Issue with accessing webcam", err);
            });
    }

    // Handle the end of a recording session
    function handleRecordingStop() {
        blob = new Blob(videoChunks, { 'type': 'video/mp4' });
        videoChunks = [];
        if (earlyStop){
            return;
        } else {
            sendData();
        }
    }

    // Function to disable and enable the session button
    function toggleButtonDisabled(state) {
        sessionBtn.disabled = state;
    }

    // Adjustments in sendData function
    function sendData() {
        toggleButtonDisabled(true); // Disable the button when POST request starts
        statusText.innerText = "Processing...";
        const duration = (Date.now() - startTime) / 1000;
    
        const formData = new FormData();
        formData.append('video', blob);
        formData.append('duration', duration.toString());
    
        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                document.getElementById('generated-art').src = `/${data.imagePath}?timestamp=${new Date().getTime()}`;
                engagementScoreElement.innerText = `Engagement Score: ${data.engagementScore}`;
                averageDurationElement.innerText = `Average Time Spent On Art: ${data.averageDuration}`;
                if (data.sessionEnd) {
                    statusText.innerText = "Session has ended. Thank you!";
                    // sessionBtn.disabled = true; 
                } else {
                    toggleButtonDisabled(false);
                    // statusText.innerText = "New art displayed. Recording will start shortly...";
                    dataSent = true;
                    setTimeout(startRecording, 0);
                }
            } else {
                console.error("Failed to process video: ", data.message);
                // statusText.innerText = "Failed to update image. Check server connection.";
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            statusText.innerText = "Error sending video.";
        })
    }
    
    submitScoreBtn.addEventListener('click', function () {
        let score = document.getElementById('scoreInput').value;
        sendScore(score);
    });  
});



document.addEventListener('DOMContentLoaded', function() {
    sendScore(0); 
});


function updateArtDisplay() {
    const artImageElement = document.getElementById('generated-art');
    artImageElement.src = '/latest_art?' + new Date().getTime();
}

function sendScore(score) {
    fetch('/submit_score', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `score=${score}`
    })
    .then(response => response.json())
    .then(data => {
        if(data.status === "success" && data.imagePath) {
            document.getElementById('generated-art').src = `/${data.imagePath}`;
            if (data.sessionEnd) {
                // document.getElementById('submitScoreBtn').disabled = true;
                document.getElementById('submitScoreBtn').innerText = "Session Ended";
                document.getElementById('status').innerText = "Session has ended. Thank you!";
            }
        } else {
            console.error("Image path not received or score submission was not successful.", data.message);
            // document.getElementById('status').innerText = "Failed to update. Check connection or data.";
        }
    })
    .catch((error) => {
        console.error('Error:', error);
        // document.getElementById('status').innerText = "Error sending score.";
    });
}
