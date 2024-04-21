document.addEventListener('DOMContentLoaded', function () {
    let video = document.getElementById('video');
    let recordBtn = document.getElementById('recordBtn');
    let submitScoreBtn = document.getElementById('submitScoreBtn');
    let statusText = document.getElementById('status');
    let mediaRecorder;
    let videoChunks = [];

    // Define the constraints for the video stream including resolution and frame rate
    const videoConstraints = {
        video: {
            width: { ideal: 1280 },  // 720p resolution
            height: { ideal: 720 },
            frameRate: { ideal: 60 }  // 60 fps
        }
    };

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia(videoConstraints)  // Use the defined constraints
            .then(function (stream) {
                video.srcObject = stream;
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = function (e) {
                    videoChunks.push(e.data);
                };

                mediaRecorder.onstop = function () {
                    let blob = new Blob(videoChunks, { 'type': 'video/mp4' });
                    videoChunks = [];
                    sendData(blob);
                    statusText.innerText = "Recording finished. Processing...";
                };
            })
            .catch(function (err) {
                console.error("Issue with accessing webcam", err);
            });
    }

    recordBtn.addEventListener('click', function () {
        if (mediaRecorder.state === "inactive") {
            mediaRecorder.start();
            statusText.innerText = "Recording in progress...";
            setTimeout(() => {
                if (mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                }
            }, 7000);
        }
    });

    submitScoreBtn.addEventListener('click', function () {
        let score = document.getElementById('scoreInput').value;
        sendScore(score);
    });
});

function sendData(videoBlob) {
    let formData = new FormData();
    formData.append('video', videoBlob, 'video.mp4');

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        document.getElementById('status').innerText = data.message;
        if (data.status === "success") {
            document.getElementById('generated-art').src = `/${data.imagePath}`;
            document.getElementById('engagement-score').innerText = `Engagement Score: ${data.engagementScore}`;
        } else {
            console.error("Image path not received or video processing was not successful.");
        }
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById('status').innerText = "Error sending video.";
    });
}



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
        } else {
            console.error("Image path not received or score submission was not successful.");
        }
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
