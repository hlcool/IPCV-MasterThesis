#-------------------------------------------------
#
# Project created by QtCreator 2017-04-25T14:51:09
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = MultithreadApplication
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += main.cpp\
        mainwindow.cpp \
    camerastream.cpp \
    cameraworker.cpp \
    peopledetector.cpp \
    DPM/dpm_cascade_detector.cpp \
    DPM/dpm_cascade.cpp \
    DPM/dpm_convolution.cpp \
    DPM/dpm_feature.cpp \
    DPM/dpm_model.cpp \
    DPM/dpm_nms.cpp \
    aboutdialog.cpp \
    ACF/ACFDetector.cpp \
    ACF/ACFFeaturePyramid.cpp \
    ACF/Channel.cpp \
    ACF/ChannelFeatures.cpp \
    ACF/ColorChannel.cpp \
    ACF/convConst.cpp \
    ACF/GradHistChannel.cpp \
    ACF/gradientMex.cpp \
    ACF/GradMagChannel.cpp \
    ACF/wrappers.cpp \
    ACF/Core/detection.cpp \
    ACF/Core/DetectionList.cpp \
    ACF/Core/detector.cpp \
    ACF/Core/DetectorManager.cpp \
    ACF/Core/detectormodel.cpp \
    ACF/Core/feature.cpp \
    ACF/Core/featurelayer.cpp \
    ACF/Core/filelocator.cpp \
    ACF/Core/FileWriter.cpp \
    ACF/Core/FrameFromCamera.cpp \
    ACF/Core/FrameFromDirectory.cpp \
    ACF/Core/FrameFromVideo.cpp \
    ACF/Core/FrameProducer.cpp \
    ACF/Core/nms.cpp \
    ACF/Core/NonMaximumSuppression.cpp \
    ACF/Core/ScaleSpacePyramid.cpp \
    ACF/Channel/Convert.cpp \
    evaluation.cpp

HEADERS  += mainwindow.h \
    barrier.h \
    camerastream.h \
    cameraworker.h \
    cvimagewidget.h \
    peopledetector.h \
    DPM/dpm_cascade.hpp \
    DPM/dpm_convolution.hpp \
    DPM/dpm_feature.hpp \
    DPM/dpm_model.hpp \
    DPM/dpm_nms.hpp \
    DPM/dpm.hpp \
    DPM/precomp.hpp \
    aboutdialog.h \
    ACF/ACFDetector.h \
    ACF/ACFFeaturePyramid.h \
    ACF/Channel.h \
    ACF/ChannelFeatures.h \
    ACF/ColorChannel.h \
    ACF/GradHistChannel.h \
    ACF/GradMagChannel.h \
    ACF/imResampleMex.hpp \
    ACF/rgbConvertMex.hpp \
    ACF/sse.hpp \
    ACF/wrappers.hpp \
    ACF/Core/detection.h \
    ACF/Core/DetectionList.h \
    ACF/Core/detector.h \
    ACF/Core/DetectorManager.h \
    ACF/Core/detectormodel.h \
    ACF/Core/dirent.h \
    ACF/Core/feature.h \
    ACF/Core/featurelayer.h \
    ACF/Core/filelocator.h \
    ACF/Core/FileWriter.h \
    ACF/Core/FrameFromCamera.h \
    ACF/Core/FrameFromDirectory.h \
    ACF/Core/FrameFromVideo.h \
    ACF/Core/FrameProducer.h \
    ACF/Core/Image.hpp \
    ACF/Core/nms.h \
    ACF/Core/NonMaximumSuppression.h \
    ACF/Core/ScaleSpacePyramid.h \
    ACF/rapidxml-1.13/rapidxml_iterators.hpp \
    ACF/rapidxml-1.13/rapidxml_print.hpp \
    ACF/rapidxml-1.13/rapidxml_utils.hpp \
    ACF/rapidxml-1.13/rapidxml.hpp \
    ACF/Channel/Functions.h \
    evaluation.h

FORMS    += mainwindow.ui \
    aboutdialog.ui

# The following lines tells Qmake to use pkg-config for opencv
QT_CONFIG -= no-pkg-config
CONFIG  += link_pkgconfig
PKGCONFIG += opencv

QMAKE_CXXFLAGS_WARN_ON += -Wno-reorder

RESOURCES += \
    resources.qrc
