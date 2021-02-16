const video = document.getElementById('webcam');
const predView = document.getElementById('prediction');

const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');

// Check if webcam access is supported.
function getUserMediaSupported() {
    return !!(navigator.mediaDevices &&
        navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam function which we will 
// define in the next step.
if (getUserMediaSupported()) {
    enableWebcamButton.addEventListener('click', enableCam);
} else {
    console.warn('getUserMedia() is not supported by your browser');
}

// Enable the live webcam view and start classification.
async function enableCam(event) {
    // // Hide the button once clicked.
    event.target.classList.add('removed');

    // // getUsermedia parameters to force video but not audio.
    const constraints = {
        video: true
    };

    // // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        // video.srcObject = stream;
        video.addEventListener('loadeddata', predictWebcam);
    });
    webcam = await tf.data.webcam(video);
}

var model = undefined;
var modelInputShape = undefined;
var webcam = undefined;
var predictions = undefined;
var frames = 0;

function timer() {
    console.log(`Total Frames: ${frames}`);
    frames = 0;
}

async function init() {
    // Store the resulting model in the global scope of our app.
    model = await tf.loadGraphModel('model.json');
    modelInputShape = model.inputs[0].shape;
    modelInputShape = [modelInputShape[1], modelInputShape[2]]
    demosSection.classList.remove('invisible');
    setInterval(timer, 1000);
}

var children = [];

async function predictWebcam() {
    console.log("Here");
    while (true) {
        // Capture the frame from the webcam.
        const img = await getImage();

        // Only Render on alternate frames
        if (frames % 2 == 0)
        {
            // Make a prediction through our newly-trained model using the embeddings
            // from mobilenet as input.
            predictions = await model.predict(img);
            let [background, person] = predictions.resizeNearestNeighbor([512, 512]).split(2, 3);
        }
        frames += 1;
        await tf.nextFrame();
    }
}

/**
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
async function getImage() {
    const img = await webcam.capture();
    const processedImg = tf.tidy(() => img.resizeNearestNeighbor(modelInputShape).expandDims(0).toFloat().div(127).sub(1));
    img.dispose();
    return processedImg;
}

init();