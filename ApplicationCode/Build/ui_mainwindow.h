/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QWidget>
#include <cvimagewidget.h>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpen_file;
    QAction *actionCloseWindow;
    QWidget *centralWidget;
    QLabel *label;
    CVImageWidget *CVWidget;
    QMenuBar *menuBar;
    QMenu *menuFile;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(746, 522);
        actionOpen_file = new QAction(MainWindow);
        actionOpen_file->setObjectName(QStringLiteral("actionOpen_file"));
        actionCloseWindow = new QAction(MainWindow);
        actionCloseWindow->setObjectName(QStringLiteral("actionCloseWindow"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        label = new QLabel(centralWidget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(289, 30, 160, 16));
        CVWidget = new CVImageWidget(centralWidget);
        CVWidget->setObjectName(QStringLiteral("CVWidget"));
        CVWidget->setGeometry(QRect(90, 80, 561, 381));
        CVWidget->setMaximumSize(QSize(800, 800));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 746, 22));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        MainWindow->setMenuBar(menuBar);

        menuBar->addAction(menuFile->menuAction());
        menuFile->addAction(actionOpen_file);
        menuFile->addAction(actionCloseWindow);

        retranslateUi(MainWindow);
        QObject::connect(actionCloseWindow, SIGNAL(triggered()), MainWindow, SLOT(close()));

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", Q_NULLPTR));
        actionOpen_file->setText(QApplication::translate("MainWindow", "Open file...", Q_NULLPTR));
        actionCloseWindow->setText(QApplication::translate("MainWindow", "Close", Q_NULLPTR));
#ifndef QT_NO_TOOLTIP
        actionCloseWindow->setToolTip(QApplication::translate("MainWindow", "CloseWindow", Q_NULLPTR));
#endif // QT_NO_TOOLTIP
        label->setText(QApplication::translate("MainWindow", "Display an OpenCV Video", Q_NULLPTR));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
