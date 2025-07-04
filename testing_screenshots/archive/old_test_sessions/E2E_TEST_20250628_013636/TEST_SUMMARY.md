# GEOTECHNICAL_ANALYSIS_TOOL_COMPLETE_FUNCTIONAL_VERIFICATION - Test Summary\n\n**Test Session:** 20250628_013636\n**Duration:** 0.2 minutes\n**Success Rate:** 8.0%\n**Screenshots:** 3\n**Plots Generated:** 0\n\n## Test Results\n- ✅ **SETUP**: Chrome driver initialized\n- 📋 **TEST_01**: Testing application startup\n- ✅ **TEST_01**: Application loaded successfully\n- 📋 **TEST_02**: Testing data upload functionality\n- ❌ **TEST_02**: Data upload failed: Message: invalid argument: File not found : /Users/qinli/Library/CloudStorage/OneDrive-CPBContractorsPtyLTD/01 Digitisation Project/Data Analysis App/testing_screenshots/Lab_summary_final.xlsx
  (Session info: chrome=137.0.7151.120)
Stacktrace:
0   chromedriver                        0x00000001028cdd14 cxxbridge1$str$ptr + 2735276
1   chromedriver                        0x00000001028c5f88 cxxbridge1$str$ptr + 2703136
2   chromedriver                        0x00000001024166f0 cxxbridge1$string$len + 90424
3   chromedriver                        0x0000000102456c74 cxxbridge1$string$len + 353980
4   chromedriver                        0x00000001024531b4 cxxbridge1$string$len + 338940
5   chromedriver                        0x000000010249f0c8 cxxbridge1$string$len + 650000
6   chromedriver                        0x0000000102451be8 cxxbridge1$string$len + 333360
7   chromedriver                        0x0000000102891598 cxxbridge1$str$ptr + 2487600
8   chromedriver                        0x0000000102894830 cxxbridge1$str$ptr + 2500552
9   chromedriver                        0x0000000102871c14 cxxbridge1$str$ptr + 2358188
10  chromedriver                        0x00000001028950b8 cxxbridge1$str$ptr + 2502736
11  chromedriver                        0x0000000102862dec cxxbridge1$str$ptr + 2297220
12  chromedriver                        0x00000001028b5420 cxxbridge1$str$ptr + 2634680
13  chromedriver                        0x00000001028b55ac cxxbridge1$str$ptr + 2635076
14  chromedriver                        0x00000001028c5bd4 cxxbridge1$str$ptr + 2702188
15  libsystem_pthread.dylib             0x0000000187a9ac0c _pthread_start + 136
16  libsystem_pthread.dylib             0x0000000187a95b80 thread_start + 8
\n- 📋 **TEST_03**: Testing Analysis Workspace navigation and tabs\n- 📋 **TAB_TEST**: Testing 📊 PSD Analysis tab\n- ❌ **TAB_TEST**: Could not find 📊 PSD Analysis tab\n- 📋 **TAB_TEST**: Testing 🧪 Atterberg Limits tab\n- ❌ **TAB_TEST**: Could not find 🧪 Atterberg Limits tab\n- 📋 **TAB_TEST**: Testing 🔨 SPT Analysis tab\n- ❌ **TAB_TEST**: Could not find 🔨 SPT Analysis tab\n- 📋 **TAB_TEST**: Testing 💪 UCS Analysis tab\n- ❌ **TAB_TEST**: Could not find 💪 UCS Analysis tab\n- 📋 **TAB_TEST**: Testing 🌍 Spatial Analysis tab\n- ❌ **TAB_TEST**: Could not find 🌍 Spatial Analysis tab\n- 📋 **TAB_TEST**: Testing 🔬 Emerson Analysis tab\n- ❌ **TAB_TEST**: Could not find 🔬 Emerson Analysis tab\n- ❌ **TEST_03**: Analysis tabs tested: 0/6 successful\n- 📋 **TEST_04**: Testing Dashboard Gallery\n- ❌ **TEST_04**: Dashboard Gallery button not found\n- 📋 **TEST_05**: Testing Export & Reporting functionality\n- ❌ **TEST_05**: Export & Reporting button not found\n- 📋 **TEST_06**: Testing Single-Page Mode\n- ⚠️ **TEST_06**: Could not find mode selector\n