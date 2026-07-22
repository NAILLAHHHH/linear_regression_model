import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

// change this to your deployed render url
const apiBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'https://linear-regression-model-api.onrender.com',
);

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Life Expectancy',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const PredictPage(),
    );
  }
}

class PredictPage extends StatefulWidget {
  const PredictPage({super.key});

  @override
  State<PredictPage> createState() => _PredictPageState();
}

class _PredictPageState extends State<PredictPage> {
  // same fields as the API
  final controllers = {
    'status': TextEditingController(),
    'adult_mortality': TextEditingController(),
    'alcohol': TextEditingController(),
    'bmi': TextEditingController(),
    'polio': TextEditingController(),
    'hiv_aids': TextEditingController(),
    'gdp': TextEditingController(),
    'income_composition': TextEditingController(),
    'schooling': TextEditingController(),
  };

  final ranges = {
    'status': [0.0, 1.0],
    'adult_mortality': [0.0, 800.0],
    'alcohol': [0.0, 20.0],
    'bmi': [1.0, 90.0],
    'polio': [0.0, 100.0],
    'hiv_aids': [0.0, 60.0],
    'gdp': [0.0, 150000.0],
    'income_composition': [0.0, 1.0],
    'schooling': [0.0, 25.0],
  };

  String result = '';
  bool loading = false;

  @override
  void dispose() {
    for (final c in controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> predict() async {
    final labels = {
      'status': 'Status',
      'adult_mortality': 'Adult mortality',
      'alcohol': 'Alcohol',
      'bmi': 'BMI',
      'polio': 'Polio %',
      'hiv_aids': 'HIV/AIDS',
      'gdp': 'GDP',
      'income_composition': 'Income composition',
      'schooling': 'Schooling years',
    };

    // collect all empty fields first
    final missing = <String>[];
    for (final key in controllers.keys) {
      if (controllers[key]!.text.trim().isEmpty) {
        missing.add(labels[key]!);
      }
    }
    if (missing.isNotEmpty) {
      setState(() => result = 'Please fill these first: ${missing.join(', ')}');
      return;
    }

    // then check numbers and ranges
    for (final key in controllers.keys) {
      final text = controllers[key]!.text.trim();
      final value = double.tryParse(text);
      if (value == null) {
        setState(() => result = 'Error: ${labels[key]} must be a number');
        return;
      }
      final min = ranges[key]![0];
      final max = ranges[key]![1];
      if (value < min || value > max) {
        setState(
          () => result = 'Error: ${labels[key]} must be between $min and $max',
        );
        return;
      }
    }

    final body = {
      for (final key in controllers.keys)
        key: key == 'status'
            ? int.parse(controllers[key]!.text.trim())
            : double.parse(controllers[key]!.text.trim()),
    };

    setState(() {
      loading = true;
      result = 'Loading...';
    });

    try {
      final response = await http
          .post(
            Uri.parse('$apiBaseUrl/predict'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          result =
              'Predicted life expectancy: ${data['predicted_life_expectancy']} years\n'
              'Model: ${data['model_name']}';
        });
      } else {
        setState(() => result = 'Error: ${response.body}');
      }
    } catch (e) {
      setState(
        () => result =
            'Could not reach the API. Check your connection and API URL.',
      );
    } finally {
      setState(() => loading = false);
    }
  }

  Widget field(String key, String label, String hint) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: TextField(
        controller: controllers[key],
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          border: const OutlineInputBorder(),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Life Expectancy Predictor')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Enter the values below then press Predict.\n'
              'status: 0 = Developing, 1 = Developed',
            ),
            const SizedBox(height: 16),
            field('status', 'Status', 'e.g. 0 = Developing, 1 = Developed'),
            field('adult_mortality', 'Adult mortality', 'e.g. 120 (0-800 per 1000)'),
            field('alcohol', 'Alcohol', 'e.g. 4.5 (litres, 0-20)'),
            field('bmi', 'BMI', 'e.g. 25 (1-90)'),
            field('polio', 'Polio %', 'e.g. 90 (0-100)'),
            field('hiv_aids', 'HIV/AIDS', 'e.g. 0.2 (0-60)'),
            field('gdp', 'GDP', 'e.g. 4500 (0-150000)'),
            field('income_composition', 'Income composition', 'e.g. 0.65 (0-1)'),
            field('schooling', 'Schooling years', 'e.g. 12 (0-25)'),
            const SizedBox(height: 8),
            ElevatedButton(
              onPressed: loading ? null : predict,
              child: const Text('Predict'),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              color: Colors.teal.shade50,
              child: Text(result.isEmpty ? 'Result will show here' : result),
            ),
          ],
        ),
      ),
    );
  }
}
