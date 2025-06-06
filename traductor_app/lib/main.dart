import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp( MaterialApp(
    home: SignRecognizer(),
  ));
}

class SignRecognizer extends StatefulWidget {
  @override
  _SignRecognizerState createState() => _SignRecognizerState();
}

class _SignRecognizerState extends State<SignRecognizer> {
  CameraController? controller;
  String letter = '';

  @override
  void initState() {
    super.initState();
    initCamera();
  }

  Future<void> initCamera() async {
    final cameras = await availableCameras();
    controller = CameraController(cameras[0], ResolutionPreset.medium);
    await controller!.initialize();
    controller!.startImageStream((CameraImage image) {
      processFrame(image);
    });
    setState(() {});
  }

  bool _sending = false;

  void processFrame(CameraImage image) async {
    if (_sending) return;
    _sending = true;

    try {
      final bytes = await convertYUV420toImage(image);
      
      final request = MultipartFile.fromBytes(
        bytes,
        filename: 'frame.png',
      );
      
      final response = await Dio().post(
        'https://traductor-api-fhon.onrender.com/predict',
        data: request,
      );
      final respStr = await response.data;
      setState(() {
        letter = RegExp(r'"letter":"(.)"').firstMatch(respStr)?.group(1) ?? '';
      });
    } finally {
      _sending = false;
    }
  }

  Future<Uint8List> convertYUV420toImage(CameraImage image) async {
    // Convierte la imagen YUV a PNG usando la librería `image`
    final width = image.width;
    final height = image.height;
    final img.Image converted = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final uvIndex = (x ~/ 2) + (y ~/ 2) * (width ~/ 2);
        final yp = image.planes[0].bytes[y * width + x];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];
        final r = (yp + 1.370705 * (vp - 128)).clamp(0, 255).toInt();
        final g = (yp - 0.337633 * (up - 128) - 0.698001 * (vp - 128)).clamp(0, 255).toInt();
        final b = (yp + 1.732446 * (up - 128)).clamp(0, 255).toInt();
        final color = img.ColorFloat16.rgb(r, g, b);
        converted.setPixel(x, y, color);
      }
    }

    final resized = img.copyResize(converted, width: 28, height: 28);
    final grayscale = img.grayscale(resized);
    return Uint8List.fromList(img.encodePng(grayscale));
  }

  @override
  Widget build(BuildContext context) {
    if (controller == null || !controller!.value.isInitialized) return Container();
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Traductor'),
      ),
      body: Column(
        children: [
          Expanded(
            child: CameraPreview(controller!),
          ),
          Text(
              'Letra: ${ letter == "" ? "Invalido" : letter }',
              style: TextStyle(fontSize: 32, color: Colors.white),
            ),
        ],
      ),
    );
  }
}
