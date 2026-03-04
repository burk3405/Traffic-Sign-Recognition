# Traffic Sign Recognition  
C++ | OpenCV 4.12 | Visual Studio 2022 | Windows x64

---

## Requirements

This project requires:

- **OpenCV 4.12.0**
- `opencv_world4120.lib` (Release)
- `opencv_world4120d.lib` (Debug)
- Visual Studio 2022 (Desktop Development with C++)

---

# Setup Guide (Visual Studio 2022 – Windows)

---

## 1️ Install OpenCV 4.12

1. Go to:  
   https://opencv.org/releases/

2. Download **OpenCV 4.12.0 – Windows**

3. Extract it to a permanent location, for example:

```
C:\opencv
```

After extracting, you should see:

```
C:\opencv\build\include
C:\opencv\build\x64\vc17\lib
C:\opencv\build\x64\vc17\bin
```

> If using Visual Studio 2019, the folder may be `vc16` instead of `vc17`.

---

## 2️ Configure Include Directory

This allows the compiler to find OpenCV headers like:

```cpp
#include <opencv2/opencv.hpp>
```

### Steps:

1. Right-click project → **Properties**
2. Set:
   - Configuration: **All Configurations**
   - Platform: **x64**
3. Navigate to:

```
C/C++ → General → Additional Include Directories
```

4. Add:

```
C:\opencv\build\include
```

---

## 3️ Configure Library Directory

This allows the linker to find `.lib` files.

1. Go to:

```
Linker → General → Additional Library Directories
```

2. Add:

For Visual Studio 2022:

```
C:\opencv\build\x64\vc17\lib
```

For Visual Studio 2019:

```
C:\opencv\build\x64\vc16\lib
```

---

## 4️ Link Required OpenCV Libraries

Go to:

```
Linker → Input → Additional Dependencies
```

### For Release Mode:
Add:
```
opencv_world4120.lib
```

### For Debug Mode:
Add:
```
opencv_world4120d.lib
```

> Make sure Debug uses the `d` version and Release does not.

---

## 5️ Add OpenCV DLLs to System PATH (Required)

If skipped, the program will compile but fail at runtime.

1. Copy:

```
C:\opencv\build\x64\vc17\bin
```

2. Press **Windows Key**
3. Search: `Environment Variables`
4. Click **Edit the system environment variables**
5. Click **Environment Variables**
6. Under **System Variables**, select `Path`
7. Click **Edit**
8. Click **New**
9. Paste the OpenCV `bin` path
10. Click OK on all dialogs

Restart Visual Studio.

---

## 6️ Ensure Platform is x64

At the top of Visual Studio, confirm the platform is:

```
x64
```

If only `Win32` appears:

1. Open **Configuration Manager**
2. Click **New Platform**
3. Select `x64`
4. Copy settings from `Win32`

---

# Test OpenCV Installation

Add this test to `main.cpp`:

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::Mat::zeros(200, 200, CV_8UC3);
    cv::imshow("Test Window", img);
    cv::waitKey(0);
    return 0;
}
```

If a black window opens, OpenCV is correctly configured.

---

# Common Errors

### LNK1104: cannot open file 'opencv_world4120.lib'
Library directory path is incorrect.

### Unresolved external symbol errors
Debug/Release library mismatch.

### Application crashes immediately after launch
OpenCV `bin` directory was not added to PATH.

---

# Configuration Checklist

- [ ] OpenCV 4.12 installed
- [ ] Include directory configured
- [ ] Library directory configured
- [ ] Correct `.lib` linked (Debug vs Release)
- [ ] OpenCV `bin` directory added to PATH
- [ ] Platform set to x64

---

## Tested With

- Windows 10 / 11
- Visual Studio 2022
- OpenCV 4.12.0
- x64 Platform
