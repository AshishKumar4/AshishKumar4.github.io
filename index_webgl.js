const video = document.getElementById('webcam');
const predView = document.getElementById('prediction');
const frameDisplay = document.getElementById('frames');

const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');

const bgCanvas = document.getElementById('bgCanvas');
const bg_ctx = bgCanvas.getContext("2d");

const webglCanvas =  document.getElementById('webgl');
const webglCanvasCtx = webglCanvas.getContext("webgl2");

const smoothFilter5 = tf.tensor4d([
    [[[1]], [[1]], [[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]], [[1]], [[1]]],
]).div(25);

const smoothFilter3 = tf.tensor4d([
    [[[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]]],
]).div(9);

var backgroundImage;

var model;
var modelInputShape;
var streamShape;
var webcam;
var predictions;
var frames = 0;
var webglMaskTexture;
var webglForegroundTexture;
var webglBackgroundTexture;

function readImage() {
    if (!this.files || !this.files[0]) return;

    const FR = new FileReader();
    FR.addEventListener("load", (evt) => {
        const img = new Image();
        img.style.objectFit = "fill";
        img.addEventListener("load", () => {
            backgroundImage = img;
            // renderWebgl(img);
            bg_ctx.drawImage(img, 0, 0);
        });
        img.src = evt.target.result;
    });
    FR.readAsDataURL(this.files[0]);
}

document.getElementById('bgUpload').addEventListener("change", readImage);

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
		const { height, width } = stream.getVideoTracks()[0].getSettings();
		streamShape = [height, width];
        video.addEventListener('loadeddata', predictWebcam);
    });
    webcam = await tf.data.webcam(video);
}

function timer() {
    frameDisplay.innerHTML = frames.toString();
    frames = 0;
}

async function predictSegmentation(img) {
    const [height, width] = streamShape;
    const [modelHeight, modelWidth] = modelInputShape;

    // const imageData = new ImageData(modelWidth, modelHeight);

    tf.tidy(() => {
        const predictions = model.predict(img).squeeze().softmax(); // .resizeBilinear(streamShape); // .softmax().log().add(1).clipByValue(0, 1)
        const [background, person] = predictions.split(2, 2);
        const segmentationMask = background;

        const segData = segmentationMask.mul(255).dataSync();

        const imageData = new Uint8ClampedArray(segData);

        predViewCtx = predView.getContext('2d');
        predViewCtx.d

        if (backgroundImage !== undefined && backgroundImage !== null) {
            predViewCtx.drawImage(backgroundImage, 0, 0,
                backgroundImage.width, backgroundImage.height,
                0, 0, width, height);
            renderWithBackgroundWebgl(imageData, video, predView);
        } else {
            renderWithBackgroundWebgl(imageData, video, null);
        }

        segmentationMask.dispose();
        background.dispose();
        person.dispose();
    });
}

async function predictWebcam() {
    console.log("Here");
    while (true) {
        // Capture the frame from the webcam.
        const img = await getImage();
        // Only Render on alternate frames
        await predictSegmentation(img);
        img.dispose();
        // raw.dispose()
        frames += 1;
        await tf.nextFrame();
    }
}

async function getImage() {
    // console.log(video, modelInputShape);
    const img = tf.tidy(() => tf.browser.fromPixels(video).resizeBilinear(modelInputShape).expandDims(0).toFloat().div(255));
    return img;
}

var result;

async function testOp() {
    const squareAndAddKernel = inputShape => ({
        variableNames: ['X'],
        outputShape: inputShape.slice(),
        userCode: `
          void main() {
              float x = getXAtOutCoords();
              float value = x * x + x;
              setOutput(value);
            }
        `
      });
      const x = tf.tensor([1, 2, 3, 4]);
      const program = squareAndAddKernel(x.shape);
      
      const info = tf.backend().compileAndRun(program, [x]);
      result = tf.engine().makeTensorFromDataId()
      console.log("===>", result, result);
}

async function init() {
    await testOp();
    // Store the resulting model in the global scope of our app.
    model = await tf.loadGraphModel('model_f16/model.json');
    modelInputShape = model.inputs[0].shape;
    modelInputShape = [modelInputShape[1], modelInputShape[2]];
    demosSection.classList.remove('invisible');
    setInterval(timer, 1000);
}

tf.ready().then(() => init());

function compileShader(gl, shaderType, shaderSource) {
    // Create the shader object
    var shader = gl.createShader(shaderType);
   
    // Set the shader source code.
    gl.shaderSource(shader, shaderSource);
   
    // Compile the shader
    gl.compileShader(shader);
   
    // Check if it compiled
    var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    if (!success) {
      // Something went wrong during compilation; get the error
      throw "could not compile shader:" + gl.getShaderInfoLog(shader);
    }
   
    return shader;
}

function createProgram(gl, vertexShader, fragmentShader) {
     // create a program.
     var program = gl.createProgram();
    
     // attach the shaders.
     gl.attachShader(program, vertexShader);
     gl.attachShader(program, fragmentShader);
    
     // link the program.
     gl.linkProgram(program);
    
     // Check if it linked.
     var success = gl.getProgramParameter(program, gl.LINK_STATUS);
     if (!success) {
         // something went wrong with the link
         throw ("program filed to link:" + gl.getProgramInfoLog (program));
     }
    
     return program;
}

function createTexture(gl, index = 0) {
    const texture = gl.createTexture();
    // make unit 0 the active texture uint
    // (ie, the unit all other texture commands will affect
    webglCanvasCtx.activeTexture(webglCanvasCtx.TEXTURE0 + index);

    // Bind it to texture unit 0' 2D bind point
    webglCanvasCtx.bindTexture(webglCanvasCtx.TEXTURE_2D, texture);

    // Set the parameters so we don't need mips and so we're not filtering
    // and we don't repeat at the edges
    webglCanvasCtx.texParameteri(webglCanvasCtx.TEXTURE_2D, webglCanvasCtx.TEXTURE_WRAP_S, webglCanvasCtx.CLAMP_TO_EDGE);
    webglCanvasCtx.texParameteri(webglCanvasCtx.TEXTURE_2D, webglCanvasCtx.TEXTURE_WRAP_T, webglCanvasCtx.CLAMP_TO_EDGE);
    webglCanvasCtx.texParameteri(webglCanvasCtx.TEXTURE_2D, webglCanvasCtx.TEXTURE_MIN_FILTER, webglCanvasCtx.LINEAR);
    webglCanvasCtx.texParameteri(webglCanvasCtx.TEXTURE_2D, webglCanvasCtx.TEXTURE_MAG_FILTER, webglCanvasCtx.LINEAR);
    
    return texture;
}

function createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer
  ) {
    const program = createProgram(gl, vertexShader, fragmentShader);
  
    const positionAttributeLocation = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionAttributeLocation);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);
  
    const texCoordAttributeLocation = gl.getAttribLocation(program, 'a_texCoord');
    gl.enableVertexAttribArray(texCoordAttributeLocation);
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.vertexAttribPointer(texCoordAttributeLocation, 2, gl.FLOAT, false, 0, 0);
  
    return program;
}

function webglImage() {
    if (!webglCanvasCtx) {
        console.log("No WEBGL supported!");
        return;
    }
    const vertexShaderSource = `#version 300 es
     
    // an attribute is an input (in) to a vertex shader.
    // It will receive data from a buffer
    in vec2 a_position;
    in vec2 a_texCoord;

    // Used to pass in the resolution of the canvas
    // uniform vec2 u_resolution;

    // Used to pass the texture coordinates to the fragment shader
    out vec2 v_texCoord;

    // all shaders have a main function
    void main() {

        // convert the position from pixels to 0.0 to 1.0
        // vec2 zeroToOne = a_position / u_resolution;

        // // convert from 0->1 to 0->2
        // vec2 zeroToTwo = zeroToOne * 2.0;

        // // convert from 0->2 to -1->+1 (clipspace)
        // vec2 clipSpace = zeroToTwo - 1.0;

        gl_Position = vec4(a_position * vec2(1, -1), 0, 1);

        // pass the texCoord to the fragment shader
        // The GPU will interpolate this value between points.
        v_texCoord = a_texCoord;
    }
    `;
     
    const fragmentShaderSource = `#version 300 es
     
    precision highp float;
    
    // our texture
    uniform sampler2D u_mask;
    uniform sampler2D u_foreground;
    uniform sampler2D u_background;

    uniform vec2 u_resolution;
    
    const float offset[3] = float[](0.0, 1.3846153846, 3.2307692308);
    const float weight[3] = float[](0.2270270270, 0.3162162162, 0.0702702703);

    const vec2 b_direction = vec2(2.0, 0.0);

    // the texCoords passed in from the vertex shader.
    in vec2 v_texCoord;
    
    // we need to declare an output for the fragment shader
    out vec4 outColor;

    vec4 blur9(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
        vec2 off1 = (vec2(1.3846153846) * direction) / resolution;
        vec2 off2 = (vec2(3.2307692308) * direction) / resolution;
        vec4 color = texture(image, uv) * 0.2270270270;
        color += texture(image, uv + off1) * 0.3162162162;
        color += texture(image, uv - off1) * 0.3162162162;
        color += texture(image, uv + off2) * 0.0702702703;
        color += texture(image, uv - off2) * 0.0702702703;
        return color;
    }

	vec4 blurBilateral(sampler2D image, vec2 uv, vec2 resolution) {
		float magnitude = 4.0;
		vec2 direction = vec2(magnitude, 0.);
        vec2 off1 = (vec2(1.3846153846) * direction) / resolution;
        vec2 off2 = (vec2(3.2307692308) * direction) / resolution;
        vec4 color = texture(image, uv) * 0.2270270270;
        color += texture(image, uv + off1) * 0.3162162162;
        color += texture(image, uv - off1) * 0.3162162162;
        color += texture(image, uv + off2) * 0.0702702703;
        color += texture(image, uv - off2) * 0.0702702703;

		direction = vec2(0., magnitude);
        off1 = (vec2(1.3846153846) * direction) / resolution;
        off2 = (vec2(3.2307692308) * direction) / resolution;
        color += texture(image, uv + off1) * 0.3162162162;
        color += texture(image, uv - off1) * 0.3162162162;
        color += texture(image, uv + off2) * 0.0702702703;
        color += texture(image, uv - off2) * 0.0702702703;
        return color;
    }

    vec4 blur5(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
        vec4 color = vec4(0.0);
        vec2 off1 = vec2(1.3333333333333333) * direction;
        color += texture(image, uv) * 0.29411764705882354;
        color += texture(image, uv + (off1 / resolution)) * 0.35294117647058826;
        color += texture(image, uv - (off1 / resolution)) * 0.35294117647058826;
        return color; 
    }

    void main() {
        // Look up a color from the texture.

        // vec2 uv = vec2(gl_FragCoord.xy / u_resolution.xy);
        // uv.y = 1.0 - uv.y;

        vec2 uv = v_texCoord;

        // float alpha = blurBilateral(u_mask, uv, u_resolution).a;
        // float alpha = blur9(u_mask, uv, u_resolution, b_direction).a;
        float alpha = texture(u_mask, v_texCoord).r;

        alpha = clamp(log(alpha) + 1.0, 0.0, 1.0);
        vec4 foregroundPixel = texture(u_foreground, uv);
        vec4 backgroundPixel = texture(u_background, uv);
        outColor = foregroundPixel + (backgroundPixel - foregroundPixel) * alpha;
    }
    `;

    const vertexShader = compileShader(webglCanvasCtx, webglCanvasCtx.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = compileShader(webglCanvasCtx, webglCanvasCtx.FRAGMENT_SHADER, fragmentShaderSource);

    // webglUtils.resizeCanvasToDisplaySize(webglCanvasCtx.canvas);
    webglCanvasCtx.viewport(0, 0, webglCanvasCtx.canvas.width, webglCanvasCtx.canvas.height);
    // Clear the canvas
    webglCanvasCtx.clearColor(0, 0, 0, 0);
    webglCanvasCtx.clear(webglCanvasCtx.COLOR_BUFFER_BIT);

    var vao = webglCanvasCtx.createVertexArray();
    webglCanvasCtx.bindVertexArray(vao);

    const positionBuffer = webglCanvasCtx.createBuffer();
    webglCanvasCtx.bindBuffer(webglCanvasCtx.ARRAY_BUFFER, positionBuffer);
    {
        const positions = [
            -1.0, -1.0,
            1.0, -1.0,
            -1.0, 1.0,
            1.0, 1.0
        ];
        webglCanvasCtx.bufferData(webglCanvasCtx.ARRAY_BUFFER, new Float32Array(positions), webglCanvasCtx.STATIC_DRAW);
    }

    const texCoordBuffer = webglCanvasCtx.createBuffer();
    webglCanvasCtx.bindBuffer(webglCanvasCtx.ARRAY_BUFFER, texCoordBuffer);
    {
        const positions = [
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0
        ];
        webglCanvasCtx.bufferData(webglCanvasCtx.ARRAY_BUFFER, new Float32Array(positions), webglCanvasCtx.STATIC_DRAW);
    }

    const program = createPiplelineStageProgram(webglCanvasCtx, vertexShader, fragmentShader, positionBuffer, texCoordBuffer);

    webglCanvasCtx.useProgram(program);

    // lookup uniforms
    const resolutionLocation = webglCanvasCtx.getUniformLocation(program, "u_resolution");
    webglCanvasCtx.uniform2f(resolutionLocation, webglCanvasCtx.canvas.width, webglCanvasCtx.canvas.height);

    // 0.0, 1.3846153846, 3.2307692308
    // 0.2270270270, 0.3162162162, 0.0702702703

    const maskLocation = webglCanvasCtx.getUniformLocation(program, "u_mask");
    const foregroundLocation = webglCanvasCtx.getUniformLocation(program, "u_foreground");
    const backgroundLocation = webglCanvasCtx.getUniformLocation(program, "u_background");
    webglCanvasCtx.uniform1i(maskLocation, 0);
    webglCanvasCtx.uniform1i(foregroundLocation, 1);
    webglCanvasCtx.uniform1i(backgroundLocation, 2);

    webglMaskTexture = createTexture(webglCanvasCtx, 0);
    webglForegroundTexture = createTexture(webglCanvasCtx, 1);
    webglBackgroundTexture = createTexture(webglCanvasCtx, 2);
}

function renderWebgl(image) {
    // Bind it to texture unit 0' 2D bind point
    webglCanvasCtx.bindTexture(webglCanvasCtx.TEXTURE_2D, webglTexture);

    // Upload the image into the texture.
    var mipLevel = 0;               // the largest mip
    var internalFormat = webglCanvasCtx.RGBA;   // format we want in the texture
    var srcFormat = webglCanvasCtx.RGBA;        // format of data we are supplying
    var srcType = webglCanvasCtx.UNSIGNED_BYTE; // type of data we are supplying
    webglCanvasCtx.texImage2D(webglCanvasCtx.TEXTURE_2D,
                    mipLevel,
                    internalFormat,
                    srcFormat,
                    srcType,
                    image);

    const primitiveType = webglCanvasCtx.TRIANGLES;
    offset = 0;
    const count = 6;
    webglCanvasCtx.drawArrays(primitiveType, offset, count);
}

function renderWithBackgroundWebgl(mask, foreground, background) {
    // Bind it to texture unit 0' 2D bind point

    // Upload the image into the texture.
    const mipLevel = 0;               // the largest mip
    const internalFormat = webglCanvasCtx.RGBA;   // format we want in the texture
    const srcFormat = webglCanvasCtx.RGBA;        // format of data we are supplying
    const srcType = webglCanvasCtx.UNSIGNED_BYTE; // type of data we are supplying

    webglCanvasCtx.bindTexture(webglCanvasCtx.TEXTURE_2D, webglMaskTexture);
    webglCanvasCtx.texImage2D(webglCanvasCtx.TEXTURE_2D,
                    mipLevel,
                    webglCanvasCtx.R8,
                    256,
                    144,
                    0,
                    webglCanvasCtx.RED,
                    srcType,
                    mask);

    webglCanvasCtx.bindTexture(webglCanvasCtx.TEXTURE_2D, webglForegroundTexture);
    webglCanvasCtx.texImage2D(webglCanvasCtx.TEXTURE_2D,
                    mipLevel,
                    internalFormat,
                    srcFormat,
                    srcType,
                    foreground);

    if (background) {
        webglCanvasCtx.bindTexture(webglCanvasCtx.TEXTURE_2D, webglBackgroundTexture);
        webglCanvasCtx.texImage2D(webglCanvasCtx.TEXTURE_2D,
                        mipLevel,
                        internalFormat,
                        srcFormat,
                        srcType,
                        background);
    }

    const primitiveType = webglCanvasCtx.TRIANGLE_STRIP;
    webglCanvasCtx.drawArrays(primitiveType, 0, 4);
}

webglImage();
  