#-------------------------------------------------
#
# Project created by QtCreator 2017-04-21T12:52:45
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = PruebasHilos
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
    cameraworker.cpp \
    peopledetector.cpp \
    DPM/dpm_cascade_detector.cpp \
    DPM/dpm_cascade.cpp \
    DPM/dpm_convolution.cpp \
    DPM/dpm_feature.cpp \
    DPM/dpm_model.cpp \
    DPM/dpm_nms.cpp \
    camerastream.cpp

HEADERS  += mainwindow.h \
    cvimagewidget.h \
    barrier.h \
    cameraworker.h \
    peopledetector.h \
    DPM/dpm_cascade.hpp \
    DPM/dpm_convolution.hpp \
    DPM/dpm_feature.hpp \
    DPM/dpm_model.hpp \
    DPM/dpm_nms.hpp \
    DPM/dpm.hpp \
    DPM/precomp.hpp \
    camerastream.h

FORMS    += mainwindow.ui

# The following lines tells Qmake to use pkg-config for opencv
QT_CONFIG -= no-pkg-config
CONFIG  += link_pkgconfig
PKGCONFIG += opencv

QMAKE_CXXFLAGS_WARN_ON += -Wno-reorder
