import 'package:bike_demand_app/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('app loads', (tester) async {
    await tester.pumpWidget(const MyApp());
    expect(find.text('Predict'), findsOneWidget);
  });
}
