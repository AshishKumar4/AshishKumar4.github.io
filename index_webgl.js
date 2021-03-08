const video = document.getElementById('webcam');
const predView = document.getElementById('prediction');
const predViewCtx = predView.getContext('2d');

const frameDisplay = document.getElementById('frames');

const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');

const bgCanvas = document.getElementById('bgCanvas');
const bg_ctx = bgCanvas.getContext("2d");

const webglCanvas = document.getElementById('webgl');
// const webglCanvas = new OffscreenCanvas(width=640, height=480);
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

var tfCanvas;
var tfCanvasCtx;

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
            const [height, width] = streamShape;
            predViewCtx.drawImage(backgroundImage, 0, 0,
                backgroundImage.width, backgroundImage.height,
                0, 0, width, height);

            // Upload the image into the texture.
            var mipLevel = 0; // the largest mip
            var internalFormat = webglCanvasCtx.RGBA; // format we want in the texture
            var srcFormat = webglCanvasCtx.RGBA; // format of data we are supplying
            var srcType = webglCanvasCtx.UNSIGNED_BYTE; // type of data we are supplying
            // webglCanvasCtx.activeTexture(webglCanvasCtx.TEXTURE2);
            webglCanvasCtx.bindTexture(webglCanvasCtx.TEXTURE_2D, webglBackgroundTexture);
            webglCanvasCtx.texImage2D(webglCanvasCtx.TEXTURE_2D,
                mipLevel,
                internalFormat,
                srcFormat,
                srcType,
                backgroundImage);
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
        const {
            height,
            width
        } = stream.getVideoTracks()[0].getSettings();
        streamShape = [height, width];
        video.addEventListener('loadeddata', predictWebcam);
    });
    tf.data.webcam(video).then((obj) => {
        webcam = obj;
        setInterval(predictWebcam, 16);
    });
}

function timer() {
    frameDisplay.innerHTML = frames.toString();
    frames = 0;
}

// async function predictSegmentation(img) {
//     const [height, width] = streamShape;
//     const [modelHeight, modelWidth] = modelInputShape;

//     // const imageData = new ImageData(modelWidth, modelHeight);
//     // const tfCanvas = tf.backend().canvas;
//     // const tfCanvasCtx = tfCanvas.getContext('webgl2');
//     // const tfCanvasCtx = webglCanvasCtx;

//     tf.tidy(() => {
//         const predictions = model.predict(img).squeeze().softmax(); // .resizeBilinear(streamShape); // .softmax().log().add(1).clipByValue(0, 1)
//         const [background, person] = predictions.split(2, 2);
//         const segmentationMask = background.mul(255);

//         const segData = segmentationMask.dataSync();

//         const imageData = new Uint8ClampedArray(segData);

//         if (backgroundImage !== undefined && backgroundImage !== null) {
//             renderWithBackgroundWebgl(imageData, video, predView);
//         } else {
//             renderWithBackgroundWebgl(imageData, video, null);
//         }

//         segmentationMask.dispose();
//         background.dispose();
//         person.dispose();
//     });
// }

async function getTensorData(tensor, tensor2) {
    const vertexSource = `#version 300 es
            precision highp float;
            in vec3 clipSpacePos;
            in vec2 uv;
            out vec2 resultUV;
            void main() {
            gl_Position = vec4(clipSpacePos, 1);
            resultUV = uv;
        }`;

    const texSource = `#version 300 es
        precision highp float;
        precision highp int;
        precision highp sampler2D;
        
        // our texture
        uniform sampler2D u_mask;
    
        in vec2 resultUV;
        
        // we need to declare an output for the fragment shader
        out vec4 outputColor;
        
        ivec2 getOutputCoords() {
          ivec2 resTexRC = ivec2(resultUV.yx *
                                 vec2(144, 256));
          int index = resTexRC.x * 256 + resTexRC.y;
          int r = index / 256;
          int c = index - r * 256;
          return ivec2(r, c);
        }

        vec2 getOutputCoords2() {
            return vec2(resultUV.yx * vec2(144, 256));
          }

        // ivec3 getOutputCoords2() {
        //     ivec2 resTexRC = ivec2(resultUV.yx *
        //                            vec2(144, 256));
        //     int index = resTexRC.x * 256 + resTexRC.y;
            
        //     return ivec3(r, c, d);
        //   }

        void main() {
            // outputColor = vec4(0.0, 1.0, 1.0, 1.0 );
            vec2 uv = getOutputCoords2();
            outputColor = texture(u_mask, uv) ;
        }
    `;

    // tfCanvas = tf.backend().canvas;
    // tfCanvasCtx = tfCanvas.getContext('webgl2');

    const segTex = tf.backend().getTexture(tensor.dataId);
    // const inTex = tf.backend().getTexture(tensor2.dataId);

    // const tensor3 = tf.relu(tf.ones([144,256]));
    // const texture = tf.backend().getTexture(tensor3.dataId);
    

    // const vertexShader = compileShader(tfCanvasCtx, tfCanvasCtx.VERTEX_SHADER, vertexSource);
    // const fragmentShader = compileShader(tfCanvasCtx, tfCanvasCtx.FRAGMENT_SHADER, texSource);

    // const program = createProgram(tfCanvasCtx, vertexShader, fragmentShader);

    // tfCanvasCtx.useProgram(program);

    // // const texture = tfCanvasCtx.createTexture();
    // // tfCanvasCtx.activeTexture(tfCanvasCtx.TEXTURE0 + 5);
    // // tfCanvasCtx.bindTexture(tfCanvasCtx.TEXTURE_2D, texture);

    // // tfCanvasCtx.texParameteri(tfCanvasCtx.TEXTURE_2D, tfCanvasCtx.TEXTURE_WRAP_S, tfCanvasCtx.CLAMP_TO_EDGE);
    // // tfCanvasCtx.texParameteri(tfCanvasCtx.TEXTURE_2D, tfCanvasCtx.TEXTURE_WRAP_T, tfCanvasCtx.CLAMP_TO_EDGE);
    // // tfCanvasCtx.texParameteri(tfCanvasCtx.TEXTURE_2D, tfCanvasCtx.TEXTURE_MIN_FILTER, tfCanvasCtx.LINEAR);
    // // tfCanvasCtx.texParameteri(tfCanvasCtx.TEXTURE_2D, tfCanvasCtx.TEXTURE_MAG_FILTER, tfCanvasCtx.LINEAR);

    // tf.webgl_util.bindColorTextureToFramebuffer(tfCanvasCtx, texture, tf.backend().gpgpu.framebuffer);

    // const maskLocation = tfCanvasCtx.getUniformLocation(program, "u_mask");
    // tfCanvasCtx.uniform1i(maskLocation, 6);
    // tfCanvasCtx.activeTexture(tfCanvasCtx.TEXTURE6);
    // tfCanvasCtx.bindTexture(tfCanvasCtx.TEXTURE_2D, segTex);

    // // tfCanvasCtx.texImage2D(tfCanvasCtx.TEXTURE_2D,
    // //     0,
    // //     tfCanvasCtx.RGBA,
    // //     tfCanvasCtx.RGBA,
    // //     tfCanvasCtx.UNSIGNED_BYTE,
    // //     video);

    dWidth = 256;
    dHeight = 144;

    // tfCanvasCtx.drawElements(tfCanvasCtx.TRIANGLE, 6, tfCanvasCtx.UNSIGNED_SHORT, 0);

    // tfCanvasCtx.finish();
    
    // tf.webgl_util.bindColorTextureToFramebuffer(tfCanvasCtx, segTex, tf.backend().gpgpu.framebuffer);
    const data = new Uint8ClampedArray(dWidth*dHeight*4); 
    await readPixelsAsync(tfCanvasCtx, 0, 0, dWidth, dHeight, tfCanvasCtx.RED, tfCanvasCtx.UNSIGNED_BYTE, data);

    // return await tensor.data();
    return data;
}

async function predictSegmentation(img) {
    const [height, width] = streamShape;
    const [modelHeight, modelWidth] = modelInputShape;

    const segmentationMask = tf.tidy(() => model.execute(img).squeeze().softmax().split(2, 2)[0]);

    const data = new Uint8ClampedArray(modelWidth*modelHeight*4); 
    await readPixelsAsync(tfCanvasCtx, 0, 0, modelWidth, modelHeight, tfCanvasCtx.RED, tfCanvasCtx.UNSIGNED_BYTE, data);
    const imageData = new Uint8ClampedArray(data);

    if (backgroundImage !== undefined && backgroundImage !== null) {
        renderWithBackgroundWebgl(imageData, video, predView);
    } else {
        renderWithBackgroundWebgl(imageData, video, null);
    }

    segmentationMask.dispose();
}

// async function predictWebcam() {
//     // console.log("Here");
//     while (true) {
//         // Capture the frame from the webcam.
//         const img = await getImage();
//         // Only Render on alternate frames
//         await predictSegmentation(img);
//         img.dispose();
//         // raw.dispose()
//         frames += 1;
//         // await tf.nextFrame();
//     }
// }

async function predictWebcam() {
    // Capture the frame from the webcam.
    const img = await getImage();
    // Only Render on alternate frames
    await predictSegmentation(img);
    img.dispose();
    // raw.dispose()
    frames += 1;
    // await tf.nextFrame();
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
    result = tf.engine().makeTensorFromDataId(info.dataId, info.shape, info.dtype);
    console.log("===>", result, result);
}

async function init() {
    await testOp();

    tfCanvas = tf.backend().canvas;
    tfCanvasCtx = tf.backend().gpgpu.gl;
    // Store the resulting model in the global scope of our app.
    model = await tf.loadGraphModel('model_f16/model.json');
    modelInputShape = model.inputs[0].shape;
    modelInputShape = [modelInputShape[1], modelInputShape[2]];
    demosSection.classList.remove('invisible');
    setInterval(timer, 1000);
}

const flags = tf.env().getFlags();
// flags.WEBGL_FORCE_F16_TEXTURES = true;
flags.WEBGL_PACK = false;
flags.WEBGL_PACK_BINARY_OPERATIONS = false;
flags.WEBGL_LAZILY_UNPACK = false;
tf.env().setFlags(flags);
tf.ready().then(() => init());

function clientWaitAsync(gl, sync, flags, interval_ms) {
    return new Promise((resolve, reject) => {
        function test() {
            const res = gl.clientWaitSync(sync, flags, 0);
            if (res == gl.WAIT_FAILED) {
                reject();
                return;
            }
            if (res == gl.TIMEOUT_EXPIRED) {
                setTimeout(test, interval_ms);
                return;
            }
            resolve();
        };
        test();
    });
}

async function getBufferSubDataAsync(
    gl, target, buffer, srcByteOffset, dstBuffer,
    /* optional */
    dstOffset, /* optional */ length) {
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();

    await clientWaitAsync(gl, sync, 0, 10);
    gl.deleteSync(sync);

    gl.bindBuffer(target, buffer);
    gl.getBufferSubData(target, srcByteOffset, dstBuffer, dstOffset, length);
    gl.bindBuffer(target, null);
}

// async function readPixelsAsync(gl, x, y, w, h, format, type, dest) {
//     const buf = gl.createBuffer();
//     gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buf);
//     gl.bufferData(gl.PIXEL_PACK_BUFFER, dest.byteLength, gl.STREAM_READ);
//     gl.readPixels(x, y, w, h, format, type, 0);
//     gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);

//     await getBufferSubDataAsync(gl, gl.PIXEL_PACK_BUFFER, buf, 0, dest);

//     gl.deleteBuffer(buf);
//     return dest;
// }


async function readPixelsAsync(gl, x, y, width, height, format, type, dest) {
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buf);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, dest.byteLength, gl.STREAM_READ);
    gl.readPixels(x, y, width, height, format, type, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  
    await getBufferSubDataAsync(gl, gl.PIXEL_PACK_BUFFER, buf, 0, dest);
  
    gl.deleteBuffer(buf);
    return dest;
  }
  
// async function getBufferSubDataAsync(gl, target, buffer, srcByteOffset, dstBuffer, dstOffset, length) {
//     const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
//     gl.flush();
//     const res = await clientWaitAsync(gl, sync);
//     gl.deleteSync(sync);
  
//     if (res !== gl.WAIT_FAILED) {
//       gl.bindBuffer(target, buffer);
//       gl.getBufferSubData(target, srcByteOffset, dstBuffer, dstOffset, length);
//       gl.bindBuffer(target, null);
//     }
// }

function clientWaitAsync(gl, sync) {
    return new Promise((resolve) => {
      function test() {
        const res = gl.clientWaitSync(sync, 0, 0);
        if (res === gl.WAIT_FAILED) {
          resolve(res);
          return;
        }
        if (res === gl.TIMEOUT_EXPIRED) {
          requestAnimationFrame(test);
          return;
        }
        resolve(res);
      }
      requestAnimationFrame(test);
    });
  }
  
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
        throw ("program filed to link:" + gl.getProgramInfoLog(program));
    }

    return program;
}

function createTexture(gl, index = 0) {
    const texture = gl.createTexture();
    // make unit 0 the active texture uint
    // (ie, the unit all other texture commands will affect
    gl.activeTexture(gl.TEXTURE0 + index);

    // Bind it to texture unit 0' 2D bind point
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Set the parameters so we don't need mips and so we're not filtering
    // and we don't repeat at the edges
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

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

const vertexShaderSource = `#version 300 es
     
    // an attribute is an input (in) to a vertex shader.
    // It will receive data from a buffer
    in vec2 a_position;
    in vec2 a_texCoord;

    // Used to pass the texture coordinates to the fragment shader
    out vec2 v_texCoord;

    // all shaders have a main function
    void main() {
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
        // outColor = foregroundPixel + (backgroundPixel - foregroundPixel) * alpha;
        outColor = backgroundPixel + (foregroundPixel - backgroundPixel) * alpha;
        // outColor = texture(u_mask, v_texCoord);
    }
`;

function webglImage() {
    if (!webglCanvasCtx) {
        console.log("No WEBGL supported!");
        return;
    }

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
    webglCanvasCtx.bindBuffer(webglCanvasCtx.ARRAY_BUFFER, positionBuffer); {
        const positions = [
            -1.0, -1.0,
            1.0, -1.0,
            -1.0, 1.0,
            1.0, 1.0
        ];
        webglCanvasCtx.bufferData(webglCanvasCtx.ARRAY_BUFFER, new Float32Array(positions), webglCanvasCtx.STATIC_DRAW);
    }

    const texCoordBuffer = webglCanvasCtx.createBuffer();
    webglCanvasCtx.bindBuffer(webglCanvasCtx.ARRAY_BUFFER, texCoordBuffer); {
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
    webglCanvasCtx.activeTexture(webglCanvasCtx.TEXTURE0);
    webglCanvasCtx.bindTexture(webglCanvasCtx.TEXTURE_2D, webglTexture);

    // Upload the image into the texture.
    var mipLevel = 0; // the largest mip
    var internalFormat = webglCanvasCtx.RGBA; // format we want in the texture
    var srcFormat = webglCanvasCtx.RGBA; // format of data we are supplying
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
    const mipLevel = 0; // the largest mip
    const internalFormat = webglCanvasCtx.RGBA; // format we want in the texture
    const srcFormat = webglCanvasCtx.RGBA; // format of data we are supplying
    const srcType = webglCanvasCtx.UNSIGNED_BYTE; // type of data we are supplying

    // webglCanvasCtx.activeTexture(webglCanvasCtx.TEXTURE0);
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

    // webglCanvasCtx.activeTexture(webglCanvasCtx.TEXTURE1);
    webglCanvasCtx.bindTexture(webglCanvasCtx.TEXTURE_2D, webglForegroundTexture);
    webglCanvasCtx.texImage2D(webglCanvasCtx.TEXTURE_2D,
        mipLevel,
        internalFormat,
        srcFormat,
        srcType,
        foreground);

    if (background) {
        // webglCanvasCtx.activeTexture(webglCanvasCtx.TEXTURE2);
        webglCanvasCtx.bindTexture(webglCanvasCtx.TEXTURE_2D, webglBackgroundTexture);
        // webglCanvasCtx.texImage2D(webglCanvasCtx.TEXTURE_2D,
        //                 mipLevel,
        //                 internalFormat,
        //                 srcFormat,
        //                 srcType,
        //                 background);
    }

    const primitiveType = webglCanvasCtx.TRIANGLE_STRIP;
    webglCanvasCtx.drawArrays(primitiveType, 0, 4);
}

webglImage();



// const vertexSource = `#version 300 es
//     precision highp float;
//     in vec3 clipSpacePos;
//     in vec2 uv;
//     out vec2 resultUV;
//     void main() {
//     gl_Position = vec4(clipSpacePos, 1);
//     resultUV = uv;
// }`;

// const vertexSource = `#version 300 es
//     precision highp float;
//     in vec2 a_position;
//     in vec2 a_texCoord;
//     out vec2 resultUV;
//     void main() {
//     gl_Position = vec4(a_position * vec2(1, -1), 0, 1);
//     resultUV = a_texCoord;
// }`;

// const texSource = `#version 300 es
//     precision highp float;
//     out vec4 outColor;
//     void main() {
//         //   float x = getXAtOutCoords();
//         //   float value = x * x + x;
//         //   setOutput(value);
//         outColor = vec4(1.0, 0.0, 1.0, 1.0);
//     }
// `;

// const segTex = tf.backend().getTexture(segmentationMask.dataId);
// tfCanvasCtx.bindFramebuffer(tfCanvasCtx.FRAMEBUFFER, null);

// // Tell WebGL how to convert from clip space to pixels
// tfCanvasCtx.viewport(0, 0, tfCanvasCtx.canvas.width, tfCanvasCtx.canvas.height);

// // Clear the canvas
// tfCanvasCtx.clearColor(0, 1, 0, 0);
// tfCanvasCtx.clear(tfCanvasCtx.COLOR_BUFFER_BIT);

// var vao = tfCanvasCtx.createVertexArray();
// tfCanvasCtx.bindVertexArray(vao);

// const positionBuffer = tfCanvasCtx.createBuffer();
// tfCanvasCtx.bindBuffer(tfCanvasCtx.ARRAY_BUFFER, positionBuffer);
// {
//     const positions = [
//         -1.0, -1.0,
//         1.0, -1.0,
//         -1.0, 1.0,
//         1.0, 1.0
//     ];
//     tfCanvasCtx.bufferData(tfCanvasCtx.ARRAY_BUFFER, new Float32Array(positions), tfCanvasCtx.STATIC_DRAW);
// }

// const texCoordBuffer = tfCanvasCtx.createBuffer();
// tfCanvasCtx.bindBuffer(tfCanvasCtx.ARRAY_BUFFER, texCoordBuffer);
// {
//     const positions = [
//         0.0, 0.0,
//         1.0, 0.0,
//         0.0, 1.0,
//         1.0, 1.0
//     ];
//     tfCanvasCtx.bufferData(tfCanvasCtx.ARRAY_BUFFER, new Float32Array(positions), tfCanvasCtx.STATIC_DRAW);
// }

// const vertexShader = compileShader(tfCanvasCtx, tfCanvasCtx.VERTEX_SHADER, vertexSource);
// const fragmentShader = compileShader(tfCanvasCtx, tfCanvasCtx.FRAGMENT_SHADER, texSource);

// const program = createPiplelineStageProgram(tfCanvasCtx, vertexShader, fragmentShader, positionBuffer, texCoordBuffer);
// // const program = createProgram(tfCanvasCtx, vertexShader, fragmentShader);

// tfCanvasCtx.useProgram(program);

// // const maskLocation = tfCanvasCtx.getUniformLocation(program, "u_mask");
// // webglCanvasCtx.uniform1i(maskLocation, 0);

// // tfCanvasCtx.activeTexture(tfCanvasCtx.TEXTURE15);
// // tfCanvasCtx.bindTexture(tfCanvasCtx.TEXTURE_2D, segTex);

// // tfCanvasCtx.drawElements(tfCanvasCtx.TRIANGLE, 6, tfCanvasCtx.UNSIGNED_SHORT, 0);
// tfCanvasCtx.drawArrays(tfCanvasCtx.TRIANGLE_STRIP, 0, 4);

// predViewCtx.drawImage(video, 0, 0);
// // tf.engine.enableScissor();
// const gpgpu = tf.backend().gpgpu;
// tfCanvasCtx.useProgram(gpgpu.program);
// tfCanvasCtx.bindBuffer(tfCanvasCtx.ELEMENT_ARRAY_BUFFER, gpgpu.indexBuffer);
// tf.gpgpu_util.bindVertexProgramAttributeStreams(tfCanvasCtx, gpgpu.program, gpgpu.vertexBuffer);