import 'package:flutter/material.dart';

void main() {
  runApp(const RPEasyApp());
}

class RPEasyApp extends StatelessWidget {
  const RPEasyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RPEasy',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1E88E5),
          background: const Color(0xFFF8F9FA), // Slightly softer off-white background
        ),
        useMaterial3: true,
        fontFamily: 'Roboto',
        appBarTheme: const AppBarTheme(
          elevation: 0,
          scrolledUnderElevation: 0,
          backgroundColor: Color(0xFFF8F9FA),
          foregroundColor: Color(0xFF1A1A1A),
          centerTitle: true,
          titleTextStyle: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.w800,
            color: Color(0xFF1A1A1A),
            letterSpacing: -0.5,
          ),
        ),
      ),
      home: const SplashScreen(),
    );
  }
}

// ==========================================
// 1. SPLASH SCREEN
// ==========================================
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(seconds: 2));
    _scaleAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.elasticOut),
    );
    _controller.forward();

    Future.delayed(const Duration(milliseconds: 3000), () {
      Navigator.of(context).pushReplacement(
        PageRouteBuilder(
          pageBuilder: (context, animation, secondaryAnimation) => const MainNavigationScreen(),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            return FadeTransition(opacity: animation, child: child);
          },
          transitionDuration: const Duration(milliseconds: 800),
        ),
      );
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: ScaleTransition(
          scale: _scaleAnimation,
          child: Image.asset('assets/logo.png', width: 250, fit: BoxFit.contain),
        ),
      ),
    );
  }
}

// ==========================================
// 2. MAIN NAVIGATION CONTAINER
// ==========================================
class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _currentIndex = 1; // Start on the Analyze tab for the demo

  final List<Widget> _pages = [
    const HomeTab(),
    const AnalyzerTab(),
    const HistoryTab(),
    const ProfileTab(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F9FA),
      appBar: AppBar(title: const Text('RPEasy')),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 450),
          // Smooth fade when switching bottom tabs
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 300),
            child: _pages[_currentIndex],
          ),
        ),
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          boxShadow: [
            BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 20, offset: const Offset(0, -5)),
          ],
        ),
        child: BottomNavigationBar(
          currentIndex: _currentIndex,
          onTap: (index) => setState(() => _currentIndex = index),
          type: BottomNavigationBarType.fixed,
          backgroundColor: Colors.white,
          elevation: 0,
          selectedItemColor: const Color(0xFF1E88E5),
          unselectedItemColor: Colors.grey.shade400,
          selectedLabelStyle: const TextStyle(fontWeight: FontWeight.w700, fontSize: 12),
          unselectedLabelStyle: const TextStyle(fontWeight: FontWeight.w500, fontSize: 12),
          items: const [
            BottomNavigationBarItem(icon: Icon(Icons.dashboard_rounded), label: 'Home'),
            BottomNavigationBarItem(icon: Icon(Icons.videocam_rounded), label: 'Analyze'),
            BottomNavigationBarItem(icon: Icon(Icons.bar_chart_rounded), label: 'History'),
            BottomNavigationBarItem(icon: Icon(Icons.person_rounded), label: 'Profile'),
          ],
        ),
      ),
    );
  }
}

// ==========================================
// 3. HOME TAB
// ==========================================
class HomeTab extends StatelessWidget {
  const HomeTab({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Welcome Back', style: TextStyle(fontSize: 16, color: Colors.grey, fontWeight: FontWeight.w600)),
          const Text('Athlete', style: TextStyle(fontSize: 32, fontWeight: FontWeight.w900, letterSpacing: -1)),
          const SizedBox(height: 30),
          
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [Color(0xFF1E88E5), Color(0xFF1565C0)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(24),
              boxShadow: [
                BoxShadow(color: const Color(0xFF1E88E5).withOpacity(0.4), blurRadius: 20, offset: const Offset(0, 10)),
              ],
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Weekly Avg RPE', style: TextStyle(color: Colors.white.withOpacity(0.8), fontSize: 14, fontWeight: FontWeight.w600)),
                    const SizedBox(height: 8),
                    const Text('7.8', style: TextStyle(color: Colors.white, fontSize: 48, fontWeight: FontWeight.w900, letterSpacing: -2)),
                  ],
                ),
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), shape: BoxShape.circle),
                  child: const Icon(Icons.trending_up_rounded, color: Colors.white, size: 32),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ==========================================
// 4. ANALYZER TAB (With Smooth Transitions)
// ==========================================
enum AppState { idle, processing, result }

class AnalyzerTab extends StatefulWidget {
  const AnalyzerTab({super.key});

  @override
  State<AnalyzerTab> createState() => _AnalyzerTabState();
}

class _AnalyzerTabState extends State<AnalyzerTab> {
  AppState _currentState = AppState.idle;
  String _selectedLift = 'Squat';
  final List<String> _lifts = ['Squat', 'Bench Press', 'Deadlift'];

  void _simulateVideoUpload() async {
    setState(() => _currentState = AppState.processing);
    await Future.delayed(const Duration(seconds: 3));
    setState(() => _currentState = AppState.result);
  }

  void _resetApp() {
    setState(() => _currentState = AppState.idle);
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 12.0),
      // This AnimatedSwitcher creates the smooth cross-fade between states
      child: AnimatedSwitcher(
        duration: const Duration(milliseconds: 500),
        switchInCurve: Curves.easeOutCubic,
        switchOutCurve: Curves.easeInCubic,
        transitionBuilder: (Widget child, Animation<double> animation) {
          return FadeTransition(
            opacity: animation,
            child: SlideTransition(
              position: Tween<Offset>(begin: const Offset(0.0, 0.05), end: Offset.zero).animate(animation),
              child: child,
            ),
          );
        },
        child: _buildBodyContent(),
      ),
    );
  }

  Widget _buildBodyContent() {
    switch (_currentState) {
      case AppState.idle:
        return _buildIdleView(key: const ValueKey('idle'));
      case AppState.processing:
        return _buildProcessingView(key: const ValueKey('processing'));
      case AppState.result:
        return _buildResultView(key: const ValueKey('result'));
    }
  }

  Widget _buildIdleView({Key? key}) {
    return Column(
      key: key,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 20),
        Image.asset('assets/logo.png', height: 100, fit: BoxFit.contain),
        const SizedBox(height: 50),
        
        const Text('Exercise Type', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Colors.grey, letterSpacing: 0.5)),
        const SizedBox(height: 12),
        
        // Premium Dropdown Design
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 10, offset: const Offset(0, 4))],
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: _selectedLift,
              isExpanded: true,
              icon: const Icon(Icons.keyboard_arrow_down_rounded, color: Color(0xFF1E88E5)),
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w600, color: Color(0xFF1A1A1A), fontFamily: 'Roboto'),
              items: _lifts.map((String lift) {
                return DropdownMenuItem(value: lift, child: Text(lift));
              }).toList(),
              onChanged: (String? newValue) => setState(() => _selectedLift = newValue!),
            ),
          ),
        ),
        
        const Spacer(),
        
        // Premium Gradient Button
        Container(
          height: 65,
          decoration: BoxDecoration(
            gradient: const LinearGradient(colors: [Color(0xFF1E88E5), Color(0xFF1565C0)]),
            borderRadius: BorderRadius.circular(20),
            boxShadow: [BoxShadow(color: const Color(0xFF1E88E5).withOpacity(0.3), blurRadius: 15, offset: const Offset(0, 8))],
          ),
          child: Material(
            color: Colors.transparent,
            child: InkWell(
              borderRadius: BorderRadius.circular(20),
              onTap: _simulateVideoUpload,
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.cloud_upload_rounded, color: Colors.white, size: 28),
                  SizedBox(width: 12),
                  Text('Upload Lift Video', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w700, letterSpacing: 0.5)),
                ],
              ),
            ),
          ),
        ),
        const SizedBox(height: 20),
      ],
    );
  }

  Widget _buildProcessingView({Key? key}) {
    return Center(
      key: key,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(color: Colors.white, shape: BoxShape.circle, boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 20)]),
            child: const CircularProgressIndicator(color: Color(0xFF1E88E5), strokeWidth: 4),
          ),
          const SizedBox(height: 40),
          const Text('Analyzing Mechanics...', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800, letterSpacing: -0.5)),
          const SizedBox(height: 12),
          Text('Extracting pose & bar velocity.', style: TextStyle(fontSize: 16, color: Colors.grey.shade600, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }

  Widget _buildResultView({Key? key}) {
    return Center(
      key: key,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: Colors.green.withOpacity(0.1), shape: BoxShape.circle),
            child: const Icon(Icons.check_rounded, color: Colors.green, size: 60),
          ),
          const SizedBox(height: 24),
          Text('$_selectedLift Analyzed', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: Colors.grey)),
          const SizedBox(height: 24),
          
          // Premium Result Card
          Container(
            padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 60),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(32),
              boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 24, offset: const Offset(0, 12))],
            ),
            child: const Column(
              children: [
                Text('Predicted RPE', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w800, color: Colors.grey, letterSpacing: 1)),
                SizedBox(height: 8),
                Text('8.5', style: TextStyle(fontSize: 72, fontWeight: FontWeight.w900, color: Color(0xFF1E88E5), letterSpacing: -3, height: 1)),
              ],
            ),
          ),
          const SizedBox(height: 40),
          
          // Insights Button
          ElevatedButton.icon(
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (context) => InsightsScreen(liftName: _selectedLift))),
            icon: const Icon(Icons.insights_rounded),
            label: const Text('View Form Breakdown'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
              backgroundColor: Colors.white,
              foregroundColor: const Color(0xFF1E88E5),
              elevation: 0,
              textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16), side: BorderSide(color: Colors.grey.shade300, width: 1.5)),
            ),
          ),
          const SizedBox(height: 20),
          TextButton(
            onPressed: _resetApp,
            child: const Text('Analyze Another Lift', style: TextStyle(color: Colors.grey, fontSize: 16, fontWeight: FontWeight.w600)),
          ),
        ],
      ),
    );
  }
}

// ==========================================
// 5. HISTORY TAB
// ==========================================
class HistoryTab extends StatelessWidget {
  const HistoryTab({super.key});

  @override
  Widget build(BuildContext context) {
    final pastLifts = [
      {'lift': 'Squat', 'rpe': '8.5', 'date': 'Today', 'icon': Icons.fitness_center_rounded},
      {'lift': 'Bench Press', 'rpe': '7.0', 'date': 'Yesterday', 'icon': Icons.horizontal_rule_rounded},
      {'lift': 'Deadlift', 'rpe': '9.0', 'date': 'Oct 24, 2025', 'icon': Icons.arrow_upward_rounded},
    ];

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 12.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Recent Sessions', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w900, letterSpacing: -0.5)),
          const SizedBox(height: 20),
          Expanded(
            child: ListView.builder(
              itemCount: pastLifts.length,
              itemBuilder: (context, index) {
                final lift = pastLifts[index];
                return Container(
                  margin: const EdgeInsets.only(bottom: 16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 10, offset: const Offset(0, 4))],
                  ),
                  child: ListTile(
                    contentPadding: const EdgeInsets.all(16),
                    leading: Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(color: const Color(0xFF1E88E5).withOpacity(0.1), borderRadius: BorderRadius.circular(12)),
                      child: Icon(lift['icon'] as IconData, color: const Color(0xFF1E88E5)),
                    ),
                    title: Text(lift['lift'] as String, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 16)),
                    subtitle: Padding(
                      padding: const EdgeInsets.only(top: 4.0),
                      child: Text(lift['date'] as String, style: TextStyle(color: Colors.grey.shade500, fontWeight: FontWeight.w500)),
                    ),
                    trailing: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        const Text('RPE', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w800, color: Colors.grey, letterSpacing: 0.5)),
                        Text(lift['rpe'] as String, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w900, color: Color(0xFF1A1A1A))),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

// ==========================================
// 6. PROFILE TAB
// ==========================================
class ProfileTab extends StatelessWidget {
  const ProfileTab({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        children: [
          const SizedBox(height: 20),
          Container(
            padding: const EdgeInsets.all(4),
            decoration: BoxDecoration(shape: BoxShape.circle, border: Border.all(color: const Color(0xFF1E88E5), width: 3)),
            child: const CircleAvatar(radius: 46, backgroundColor: Color(0xFF1E88E5), child: Icon(Icons.person_rounded, size: 40, color: Colors.white)),
          ),
          const SizedBox(height: 20),
          const Text('Athlete', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w900)),
          Text('athlete@rpeasy.com', style: TextStyle(color: Colors.grey.shade600, fontWeight: FontWeight.w500)),
          const SizedBox(height: 40),
          
          Container(
            decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(20), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 10)]),
            child: Column(
              children: [
                ListTile(
                  contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                  leading: const Icon(Icons.settings_rounded, color: Color(0xFF1A1A1A)),
                  title: const Text('Account Settings', style: TextStyle(fontWeight: FontWeight.w600)),
                  trailing: const Icon(Icons.chevron_right_rounded, color: Colors.grey),
                  onTap: () {},
                ),
                Divider(height: 1, color: Colors.grey.shade200, indent: 20, endIndent: 20),
                ListTile(
                  contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                  leading: const Icon(Icons.cloud_done_rounded, color: Color(0xFF1E88E5)),
                  title: const Text('Cloud Processing API', style: TextStyle(fontWeight: FontWeight.w600)),
                  trailing: Switch(value: true, onChanged: (val) {}, activeColor: const Color(0xFF1E88E5)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ==========================================
// 7. LIFTING INSIGHTS
// ==========================================
class InsightsScreen extends StatelessWidget {
  final String liftName;
  const InsightsScreen({super.key, required this.liftName});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F9FA),
      appBar: AppBar(title: Text('$liftName Breakdown')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              height: 220,
              width: double.infinity,
              decoration: BoxDecoration(
                color: const Color(0xFF1A1A1A),
                borderRadius: BorderRadius.circular(24),
                boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 20, offset: const Offset(0, 10))],
              ),
              child: const Stack(
                alignment: Alignment.center,
                children: [
                  Icon(Icons.play_circle_fill_rounded, color: Colors.white54, size: 70),
                  Positioned(bottom: 16, left: 20, child: Text('Bounding Boxes & Keypoints Rendered', style: TextStyle(color: Colors.white70, fontSize: 12, fontWeight: FontWeight.w600))),
                ],
              ),
            ),
            const SizedBox(height: 40),
            
            const Text('Mechanics Output', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w900, letterSpacing: -0.5)),
            const SizedBox(height: 16),

            _buildMetricCard(icon: Icons.speed_rounded, title: 'Bar Speed (MMDetection)', value: '0.24 m/s', status: 'Slight Velocity Drop', statusColor: Colors.orange),
            _buildMetricCard(icon: Icons.accessibility_new_rounded, title: 'Pose Estimation (MMPose)', value: '80° Knee Angle', status: 'Full Range of Motion', statusColor: Colors.green),
            _buildMetricCard(icon: Icons.face_rounded, title: 'Facial Strain (OpenFace)', value: 'AU04, AU07 Spiked', status: 'High Effort Detected', statusColor: Colors.redAccent),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricCard({required IconData icon, required String title, required String value, required String status, required Color statusColor}) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 10, offset: const Offset(0, 4))],
      ),
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: const Color(0xFF1E88E5).withOpacity(0.1), borderRadius: BorderRadius.circular(16)),
              child: Icon(icon, color: const Color(0xFF1E88E5), size: 28),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: TextStyle(fontWeight: FontWeight.w700, fontSize: 13, color: Colors.grey.shade600)),
                  const SizedBox(height: 4),
                  Text(value, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w900, color: Color(0xFF1A1A1A))),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(color: statusColor.withOpacity(0.1), borderRadius: BorderRadius.circular(12)),
              child: Text(status, style: TextStyle(color: statusColor, fontWeight: FontWeight.w800, fontSize: 11, letterSpacing: 0.2)),
            ),
          ],
        ),
      ),
    );
  }
}