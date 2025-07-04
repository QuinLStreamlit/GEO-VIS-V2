# GEOTECHNICAL_ANALYSIS_TOOL_COMPLETE_FUNCTIONAL_VERIFICATION - Test Summary\n\n**Test Session:** 20250628_013136\n**Duration:** 0.1 minutes\n**Success Rate:** 7.7%\n**Screenshots:** 2\n**Plots Generated:** 0\n\n## Test Results\n- ✅ **SETUP**: Chrome driver initialized\n- 📋 **TEST_01**: Testing application startup\n- ❌ **TEST_01**: Application failed to load properly\n- 📋 **TEST_02**: Testing data upload functionality\n- ❌ **TEST_02**: Data upload failed: Message: no such element: Unable to locate element: {"method":"css selector","selector":"input[type='file']"}
  (Session info: chrome=137.0.7151.120); For documentation on this error, please visit: https://www.selenium.dev/documentation/webdriver/troubleshooting/errors#no-such-element-exception
Stacktrace:
0   chromedriver                        0x0000000101499d14 cxxbridge1$str$ptr + 2735276
1   chromedriver                        0x0000000101491f88 cxxbridge1$str$ptr + 2703136
2   chromedriver                        0x0000000100fe26f0 cxxbridge1$string$len + 90424
3   chromedriver                        0x00000001010299e0 cxxbridge1$string$len + 381992
4   chromedriver                        0x000000010106b0c8 cxxbridge1$string$len + 650000
5   chromedriver                        0x000000010101dbe8 cxxbridge1$string$len + 333360
6   chromedriver                        0x000000010145d598 cxxbridge1$str$ptr + 2487600
7   chromedriver                        0x0000000101460830 cxxbridge1$str$ptr + 2500552
8   chromedriver                        0x000000010143dc14 cxxbridge1$str$ptr + 2358188
9   chromedriver                        0x00000001014610b8 cxxbridge1$str$ptr + 2502736
10  chromedriver                        0x000000010142edec cxxbridge1$str$ptr + 2297220
11  chromedriver                        0x0000000101481420 cxxbridge1$str$ptr + 2634680
12  chromedriver                        0x00000001014815ac cxxbridge1$str$ptr + 2635076
13  chromedriver                        0x0000000101491bd4 cxxbridge1$str$ptr + 2702188
14  libsystem_pthread.dylib             0x0000000187a9ac0c _pthread_start + 136
15  libsystem_pthread.dylib             0x0000000187a95b80 thread_start + 8
\n- 📋 **TEST_03**: Testing Analysis Workspace navigation and tabs\n- ❌ **TEST_03**: Analysis Workspace button not found\n- 📋 **TEST_04**: Testing Dashboard Gallery\n- ❌ **TEST_04**: Dashboard Gallery button not found\n- 📋 **TEST_05**: Testing Export & Reporting functionality\n- ❌ **TEST_05**: Export & Reporting button not found\n- 📋 **TEST_06**: Testing Single-Page Mode\n- ⚠️ **TEST_06**: Could not find mode selector\n