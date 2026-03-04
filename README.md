# Traffic-Sign-Recognition
Computer Vision project on detecting and recognizing Traffic Signs

Building the Project with OpenCV 4.12 (Visual Studio 2022, Windows)

This project requires OpenCV 4.12 and specifically the following libraries:

opencv_world4120.lib (Release)

opencv_world4120d.lib (Debug)

Follow the steps below to properly configure Visual Studio.

1. Install OpenCV 4.12

Go to:
https://opencv.org/releases/

Download OpenCV 4.12.0 – Windows

Extract it to a permanent location, for example:

C:\opencv

After extracting, you should see:

C:\opencv\build\include
C:\opencv\build\x64\vc17\lib
C:\opencv\build\x64\vc17\bin

If using Visual Studio 2022, the folder will be vc17.
If using Visual Studio 2019, it may be vc16.

2. Configure Include Directory

This allows the compiler to find OpenCV headers such as:

#include <opencv2/opencv.hpp>
Steps:

Right-click your project → Properties

Set:

Configuration: All Configurations

Platform: x64

Navigate to:

C/C++ → General → Additional Include Directories

Add:

C:\opencv\build\include

Click OK.

3. Configure Library Directory

This allows the linker to find the .lib files.

In Project Properties, go to:

Linker → General → Additional Library Directories

Add:

For Visual Studio 2022:

C:\opencv\build\x64\vc17\lib

For Visual Studio 2019:

C:\opencv\build\x64\vc16\lib

Click OK.

4. Link the Required OpenCV Libraries

Now specify which OpenCV libraries to use.

Go to:

Linker → Input → Additional Dependencies
For Release Mode:

Add:

opencv_world4120.lib
For Debug Mode:

Add:

opencv_world4120d.lib

Make sure you add the correct library under the correct configuration (Debug vs Release).

5. Add OpenCV DLLs to System PATH (Required for Runtime)

If this step is skipped, the program may compile but fail at runtime.

Copy the following path:

C:\opencv\build\x64\vc17\bin

Press Windows Key

Search: Environment Variables

Click: Edit the system environment variables

Click Environment Variables

Under System Variables, select Path

Click Edit

Click New

Paste the OpenCV bin path

Click OK on all dialogs

Restart Visual Studio after this step.

6. Ensure Platform is x64

At the top of Visual Studio, confirm the build platform is:

x64

If only Win32 appears:

Open Configuration Manager

Click New Platform

Select x64

Copy settings from Win32

7. Test OpenCV Installation

You can verify everything is configured correctly using this minimal test:

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::Mat::zeros(200, 200, CV_8UC3);
    cv::imshow("Test Window", img);
    cv::waitKey(0);
    return 0;
}

If a black window opens, OpenCV is correctly configured.

Common Errors
LNK1104: cannot open file 'opencv_world4120.lib'

Library directory path is incorrect.

Unresolved external symbol errors

Debug/Release library mismatch.

Application crashes immediately after launch

OpenCV bin directory was not added to the system PATH.

Configuration Checklist

OpenCV 4.12 installed

Include directory configured

Library directory configured

Correct .lib linked for Debug/Release

OpenCV bin directory added to PATH

Platform set to x64

This project is configured and tested using:

Visual Studio 2022

OpenCV 4.12.0

Windows x64
