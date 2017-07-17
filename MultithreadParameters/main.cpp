#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    if (argc < 5) { // We expect 5 arguments: the program name, the source path and the destination path
        std::cerr << "Usage: " << argv[0] << ". Not enough arguments for the program" << std::endl;
        return 1;
    }

    String ParamPDDetector = argv[1];
    double ParamPDThreshold = stod(argv[2]);
    bool ParamSemanticFitlering, ParamMultiCamera;
    istringstream(argv[3]) >> ParamSemanticFitlering;
    istringstream(argv[4]) >> ParamMultiCamera;

    MainWindow w(ParamPDDetector, ParamPDThreshold, ParamSemanticFitlering, ParamMultiCamera);

    return a.exec();
}
