import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';
import 'services/rpe_api_service.dart';

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
  final ImagePicker _picker = ImagePicker();
  
  // API response data
  Map<String, dynamic>? _predictionData;
  String? _errorMessage;

  Future<void> _pickAndUploadVideo() async {
    try {
      setState(() {
        _errorMessage = null;
      });

      // Pick video from gallery
      final XFile? videoFile = await _picker.pickVideo(
        source: ImageSource.gallery,
        maxDuration: const Duration(minutes: 5),
      );

      if (videoFile == null) {
        // User cancelled
        return;
      }


      // Prompt for start and end time, with 'Use entire video' checkbox
      final times = await showDialog<Map<String, double>?>(
        context: context,
        builder: (context) {
          final startController = TextEditingController();
          final endController = TextEditingController();
          bool useEntireVideo = true;
          return StatefulBuilder(
            builder: (context, setState) {
              return AlertDialog(
                title: const Text('Crop Video'),
                content: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CheckboxListTile(
                      value: useEntireVideo,
                      onChanged: (val) {
                        setState(() => useEntireVideo = val ?? true);
                      },
                      title: const Text('Use entire video'),
                      controlAffinity: ListTileControlAffinity.leading,
                    ),
                    TextField(
                      controller: startController,
                      enabled: !useEntireVideo,
                      keyboardType: TextInputType.numberWithOptions(decimal: true),
                      decoration: const InputDecoration(
                        labelText: 'Start Time (seconds)',
                      ),
                    ),
                    TextField(
                      controller: endController,
                      enabled: !useEntireVideo,
                      keyboardType: TextInputType.numberWithOptions(decimal: true),
                      decoration: const InputDecoration(
                        labelText: 'End Time (seconds)',
                      ),
                    ),
                  ],
                ),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Cancel'),
                  ),
                  TextButton(
                    onPressed: () {
                      if (useEntireVideo) {
                        Navigator.pop(context, null);
                      } else {
                        final start = double.tryParse(startController.text);
                        final end = double.tryParse(endController.text);
                        if (start != null && end != null && end > start) {
                          Navigator.pop(context, {'start': start, 'end': end});
                        }
                      }
                    },
                    child: const Text('OK'),
                  ),
                ],
              );
            },
          );
        },
      );

      if (times == null) {
        // User selected 'use entire video' or cancelled
        setState(() => _currentState = AppState.processing);
        try {
          final result = await RPEApiService.predictRPE(
            videoPath: videoFile.path,
            liftType: _selectedLift,
          );
          setState(() {
            _predictionData = result;
            _currentState = AppState.result;
          });
        } catch (e) {
          setState(() {
            _errorMessage = e.toString();
            _currentState = AppState.idle;
          });
          if (mounted) {
            showDialog(
              context: context,
              builder: (context) => AlertDialog(
                title: const Text('Prediction Failed'),
                content: Text(_errorMessage ?? 'Unknown error'),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('OK'),
                  ),
                ],
              ),
            );
          }
        }
        return;
      }

      setState(() => _currentState = AppState.processing);

      try {
        final result = await RPEApiService.predictRPE(
          videoPath: videoFile.path,
          liftType: _selectedLift,
          startTime: times['start'],
          endTime: times['end'],
        );
        setState(() {
          _predictionData = result;
          _currentState = AppState.result;
        });
      } catch (e) {
        setState(() {
          _errorMessage = e.toString();
          _currentState = AppState.idle;
        });
        if (mounted) {
          showDialog(
            context: context,
            builder: (context) => AlertDialog(
              title: const Text('Prediction Failed'),
              content: Text(_errorMessage ?? 'Unknown error'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('OK'),
                ),
              ],
            ),
          );
        }
      }
    } catch (e) {
      print('Error picking video: $e');
      setState(() {
        _errorMessage = 'Failed to pick video: $e';
      });
    }
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
              onTap: _pickAndUploadVideo,
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
    // Extract data from API response
    final rpe = _predictionData?['predicted_rpe']?.toString() ?? '0.0';
    final repCount = _predictionData?['features']?['bar_speed']?['rep_count'] ?? 0;
    
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
          const SizedBox(height: 8),
          Text('$repCount reps detected', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.grey.shade500)),
          const SizedBox(height: 24),
          
          // Premium Result Card
          Container(
            padding: const EdgeInsets.symmetric(vertical: 40, horizontal: 60),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(32),
              boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 24, offset: const Offset(0, 12))],
            ),
            child: Column(
              children: [
                const Text('Predicted RPE', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w800, color: Colors.grey, letterSpacing: 1)),
                const SizedBox(height: 8),
                Text(rpe, style: const TextStyle(fontSize: 72, fontWeight: FontWeight.w900, color: Color(0xFF1E88E5), letterSpacing: -3, height: 1)),
                const SizedBox(height: 12),
              ],
            ),
          ),
          const SizedBox(height: 40),
          
          // Insights Button
          ElevatedButton.icon(
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (context) => InsightsScreen(liftName: _selectedLift, predictionData: _predictionData))),
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
class HistoryTab extends StatefulWidget {
  const HistoryTab({super.key});

  @override
  State<HistoryTab> createState() => _HistoryTabState();
}

class _HistoryTabState extends State<HistoryTab> {
  bool _loading = true;
  String? _error;
  List<Map<String, dynamic>> _pastLifts = [];

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final records = await RPEApiService.fetchHistory(limit: 100);
      if (!mounted) return;

      setState(() {
        _pastLifts = records;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  IconData _iconForLift(String liftType) {
    switch (liftType.toLowerCase()) {
      case 'squat':
        return Icons.fitness_center_rounded;
      case 'bench press':
        return Icons.horizontal_rule_rounded;
      case 'deadlift':
        return Icons.arrow_upward_rounded;
      default:
        return Icons.sports_gymnastics_rounded;
    }
  }

  String _formatDate(String? isoTimestamp) {
    if (isoTimestamp == null || isoTimestamp.isEmpty) {
      return 'Unknown date';
    }

    final date = DateTime.tryParse(isoTimestamp)?.toLocal();
    if (date == null) return 'Unknown date';

    final now = DateTime.now();
    final days = DateTime(now.year, now.month, now.day)
        .difference(DateTime(date.year, date.month, date.day))
        .inDays;

    if (days == 0) return 'Today';
    if (days == 1) return 'Yesterday';

    const monthNames = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    return '${monthNames[date.month - 1]} ${date.day}, ${date.year}';
  }

  Future<void> _confirmAndClearHistory() async {
    if (_loading) return;

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear History'),
        content: const Text('Remove all saved RPE history entries? This cannot be undone.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Clear'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      await RPEApiService.clearHistory();
      if (!mounted) return;
      await _loadHistory();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 12.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Recent Sessions', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w900, letterSpacing: -0.5)),
              TextButton.icon(
                onPressed: _pastLifts.isEmpty || _loading ? null : _confirmAndClearHistory,
                icon: const Icon(Icons.delete_outline_rounded, size: 18),
                label: const Text('Clear'),
                style: TextButton.styleFrom(
                  foregroundColor: Colors.red.shade400,
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator(color: Color(0xFF1E88E5)))
                : _error != null
                    ? Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text('Failed to load history', style: TextStyle(fontWeight: FontWeight.w700, color: Colors.grey.shade700)),
                            const SizedBox(height: 8),
                            Text(
                              _error!,
                              textAlign: TextAlign.center,
                              style: TextStyle(color: Colors.grey.shade500, fontSize: 12),
                            ),
                            const SizedBox(height: 12),
                            TextButton(onPressed: _loadHistory, child: const Text('Retry')),
                          ],
                        ),
                      )
                    : _pastLifts.isEmpty
                        ? Center(
                            child: Text(
                              'No lifts yet. Upload a video to start your history.',
                              textAlign: TextAlign.center,
                              style: TextStyle(color: Colors.grey.shade600, fontWeight: FontWeight.w600),
                            ),
                          )
                        : RefreshIndicator(
                            color: const Color(0xFF1E88E5),
                            onRefresh: _loadHistory,
                            child: ListView.builder(
                              itemCount: _pastLifts.length,
                              itemBuilder: (context, index) {
                                final lift = _pastLifts[index];
                                final liftType = (lift['lift_type'] ?? 'Unknown').toString();
                                final rpe = lift['predicted_rpe']?.toString() ?? 'N/A';
                                final date = _formatDate(lift['timestamp']?.toString());

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
                                      child: Icon(_iconForLift(liftType), color: const Color(0xFF1E88E5)),
                                    ),
                                    title: Text(liftType, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 16)),
                                    subtitle: Padding(
                                      padding: const EdgeInsets.only(top: 4.0),
                                      child: Text(date, style: TextStyle(color: Colors.grey.shade500, fontWeight: FontWeight.w500)),
                                    ),
                                    trailing: Column(
                                      mainAxisAlignment: MainAxisAlignment.center,
                                      crossAxisAlignment: CrossAxisAlignment.end,
                                      children: [
                                        const Text('RPE', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w800, color: Colors.grey, letterSpacing: 0.5)),
                                        Text(rpe, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w900, color: Color(0xFF1A1A1A))),
                                      ],
                                    ),
                                  ),
                                );
                              },
                            ),
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
class OverlayVideoPanel extends StatefulWidget {
  final String? videoUrl;
  const OverlayVideoPanel({super.key, required this.videoUrl});

  @override
  State<OverlayVideoPanel> createState() => _OverlayVideoPanelState();
}

class _OverlayVideoPanelState extends State<OverlayVideoPanel> {
  VideoPlayerController? _controller;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initializeVideo();
  }

  @override
  void didUpdateWidget(covariant OverlayVideoPanel oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.videoUrl != widget.videoUrl) {
      _disposeController();
      _initializeVideo();
    }
  }

  Future<void> _initializeVideo() async {
    final url = widget.videoUrl;
    if (url == null || url.isEmpty) {
      setState(() {
        _error = null;
      });
      return;
    }

    try {
      final controller = VideoPlayerController.networkUrl(Uri.parse(url));
      await controller.initialize();
      await controller.setLooping(true);
      await controller.setVolume(0);
      await controller.play();

      if (!mounted) {
        controller.dispose();
        return;
      }

      setState(() {
        _controller = controller;
        _error = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'Overlay URL was returned, but the video could not be played.';
      });
    }
  }

  void _disposeController() {
    _controller?.dispose();
    _controller = null;
  }

  @override
  void dispose() {
    _disposeController();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_controller != null && _controller!.value.isInitialized) {
      return ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: Stack(
          fit: StackFit.expand,
          children: [
            FittedBox(
              fit: BoxFit.cover,
              child: SizedBox(
                width: _controller!.value.size.width,
                height: _controller!.value.size.height,
                child: VideoPlayer(_controller!),
              ),
            ),
            Positioned(
              bottom: 14,
              left: 16,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.45),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Text(
                  'OpenFace Overlay',
                  style: TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w700),
                ),
              ),
            ),
          ],
        ),
      );
    }

    return Stack(
      alignment: Alignment.center,
      children: [
        const Icon(Icons.play_circle_fill_rounded, color: Colors.white54, size: 70),
        Positioned(
          bottom: 16,
          left: 20,
          right: 20,
          child: Text(
            _error ??
                (widget.videoUrl == null || widget.videoUrl!.isEmpty
                    ? 'No OpenFace overlay URL was returned by the backend for this lift.'
                    : 'Loading OpenFace overlay...'),
            style: const TextStyle(color: Colors.white70, fontSize: 12, fontWeight: FontWeight.w600),
            textAlign: TextAlign.center,
          ),
        ),
      ],
    );
  }
}

class InsightsScreen extends StatelessWidget {
  final String liftName;
  final Map<String, dynamic>? predictionData;
  const InsightsScreen({super.key, required this.liftName, this.predictionData});

  double? _toDouble(dynamic value) {
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value);
    return null;
  }

  String _formatInt(dynamic value, {String fallback = 'N/A'}) {
    if (value is num) return value.toInt().toString();
    if (value is String) {
      final parsed = int.tryParse(value);
      if (parsed != null) return parsed.toString();
    }
    return fallback;
  }

  String _formatNumber(dynamic value, {int decimals = 2, String fallback = 'N/A'}) {
    final numeric = _toDouble(value);
    if (numeric == null) return fallback;
    return numeric.toStringAsFixed(decimals);
  }

  String _formatSeconds(dynamic value, {int decimals = 2, String fallback = 'N/A'}) {
    final numeric = _toDouble(value);
    if (numeric == null) return fallback;
    return '${numeric.toStringAsFixed(decimals)}s';
  }

  String _formatPercent(dynamic value, {int decimals = 1, String fallback = 'N/A'}) {
    final numeric = _toDouble(value);
    if (numeric == null) return fallback;
    return '${(numeric * 100).toStringAsFixed(decimals)}%';
  }

  List<String> _topAUSummary(Map<String, dynamic>? facial, {int count = 3}) {
    if (facial == null) return [];

    final candidates = <MapEntry<String, double>>[];
    facial.forEach((key, value) {
      final parsed = _toDouble(value);
      final isAuMax = key.contains('AU') && (key.endsWith('_max') || key.endsWith('_r_max'));
      if (parsed != null && isAuMax) {
        candidates.add(MapEntry(key, parsed));
      }
    });

    candidates.sort((a, b) => b.value.compareTo(a.value));
    final top = candidates.take(count);

    return top.map((entry) {
      final auLabel = entry.key.split('_').first;
      return '$auLabel ${entry.value.toStringAsFixed(2)}';
    }).toList();
  }

  Map<String, dynamic> _barStatus(dynamic repCount, dynamic fatigueSeconds) {
    final reps = _toDouble(repCount);
    final fatigue = _toDouble(fatigueSeconds);

    if (reps == null || reps <= 0) {
      return {'label': 'Unavailable', 'color': Colors.grey};
    }

    if (fatigue == null) {
      return {'label': 'Detected', 'color': Colors.green};
    }

    if (fatigue <= 0.20) {
      return {'label': 'Stable Tempo', 'color': Colors.green};
    }
    if (fatigue <= 0.45) {
      return {'label': 'Mild Fatigue', 'color': Colors.orange};
    }
    return {'label': 'High Fatigue', 'color': Colors.red};
  }

  Map<String, dynamic> _postureStatus(dynamic dValue) {
    final postureMetric = _toDouble(dValue);
    if (postureMetric == null) {
      return {'label': 'Unavailable', 'color': Colors.grey};
    }

    if (postureMetric < 0.80) {
      return {'label': 'Stable Form', 'color': Colors.green};
    }
    if (postureMetric < 1.40) {
      return {'label': 'Form Drift', 'color': Colors.orange};
    }
    return {'label': 'High Deviation', 'color': Colors.red};
  }

  Map<String, dynamic> _facialStatus(dynamic detectionRate, dynamic confidenceMean) {
    final detection = _toDouble(detectionRate);
    final confidence = _toDouble(confidenceMean);

    if (detection == null && confidence == null) {
      return {'label': 'No Face Data', 'color': Colors.grey, 'message': 'OpenFace could not extract enough valid frames.'};
    }

    final detectionValue = detection ?? 0;
    final confidenceValue = confidence ?? 0;

    if (detectionValue >= 0.75 && confidenceValue >= 0.70) {
      return {
        'label': 'Strong Face Tracking',
        'color': Colors.green,
        'message': 'Facial strain metrics are based on consistent face detection.'
      };
    }
    if (detectionValue >= 0.50 && confidenceValue >= 0.50) {
      return {
        'label': 'Partial Face Tracking',
        'color': Colors.orange,
        'message': 'Facial metrics are usable, but some frames had missed or weaker face detection.'
      };
    }
    return {
      'label': 'Weak Face Tracking',
      'color': Colors.red,
      'message': 'Facial outputs may be noisy because the face was not tracked reliably.'
    };
  }

  @override
  Widget build(BuildContext context) {
    final barSpeed = predictionData?['features']?['bar_speed'] as Map<String, dynamic>?;
    final posture = predictionData?['features']?['posture'] as Map<String, dynamic>?;
    final facial = predictionData?['features']?['facial'] as Map<String, dynamic>?;

    final barRepCount = barSpeed?['rep_count'];
    final barFirstRep = barSpeed?['first_rep_duration_s'];
    final barLastRep = barSpeed?['last_rep_duration_s'];
    final barFatigue = barSpeed?['fatigue_s'];

    final postureDValue = posture?['d_value'];
    final postureRepCount = posture?['rep_count'];

    final facialDetection = facial?['detection_rate'];
    final facialConfidence = facial?['confidence_mean'];
    final topAUs = _topAUSummary(facial);
    final openfaceOverlayUrl = predictionData?['openface_overlay_url']?.toString();

    final barStatus = _barStatus(barRepCount, barFatigue);
    final postureStatus = _postureStatus(postureDValue);
    final facialStatus = _facialStatus(facialDetection, facialConfidence);

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
              child: OverlayVideoPanel(
                videoUrl: openfaceOverlayUrl,
              ),
            ),
            const SizedBox(height: 40),
            
            const Text('Mechanics Output', style: TextStyle(fontSize: 20, fontWeight: FontWeight.w900, letterSpacing: -0.5)),
            const SizedBox(height: 16),

            _buildMetricCard(
              icon: Icons.speed_rounded, 
              title: 'Bar Speed', 
              value: _formatInt(barRepCount),
              valueLabel: 'reps',
              status: barStatus['label'], 
              statusColor: barStatus['color'],
              details: [
                'First rep duration: ${_formatSeconds(barFirstRep)}',
                'Last rep duration: ${_formatSeconds(barLastRep)}',
                'Fatigue delta: ${_formatSeconds(barFatigue)}',
              ],
            ),
            _buildMetricCard(
              icon: Icons.accessibility_new_rounded, 
              title: 'Pose Estimation (MMPose)', 
              value: _formatNumber(postureDValue),
              valueLabel: 'D-metric',
              status: postureStatus['label'], 
              statusColor: postureStatus['color'],
              details: [
                'Reps used for posture metric: ${_formatInt(postureRepCount)}',
                'Higher D-metric generally means more form deviation.',
              ],
            ),
            _buildMetricCard(
              icon: Icons.face_rounded, 
              title: 'Facial Strain (OpenFace)', 
              value: _formatPercent(facialDetection),
              valueLabel: '% detected',
              status: facialStatus['label'], 
              statusColor: facialStatus['color'],
              details: [
                'Face-tracking confidence: ${_formatPercent(facialConfidence)}',
                facialStatus['message'],
                'Top AU peaks: ${topAUs.isEmpty ? 'N/A' : topAUs.join(' • ')}',
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricCard({required IconData icon, required String title, required String value, String? valueLabel, required String status, required Color statusColor, List<String> details = const []}) {
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
                  Row(
                    children: [
                      Text(value, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w900, color: Color(0xFF1A1A1A))),
                      if (valueLabel != null) ...[
                        const SizedBox(width: 4),
                        Text(valueLabel, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Colors.grey.shade500)),
                      ],
                    ],
                  ),
                  if (details.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    ...details.map(
                      (line) => Padding(
                        padding: const EdgeInsets.only(bottom: 2),
                        child: Text(
                          line,
                          style: TextStyle(fontSize: 12, color: Colors.grey.shade600, fontWeight: FontWeight.w500),
                        ),
                      ),
                    ),
                  ],
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