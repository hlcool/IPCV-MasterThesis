#ifndef POINTSSELECTWINDOW_H
#define POINTSSELECTWINDOW_H

#include <QDialog>

namespace Ui {
class pointsselectwindow;
}

class pointsselectwindow : public QDialog
{
    Q_OBJECT

public:
    explicit pointsselectwindow(QWidget *parent = 0);
    ~pointsselectwindow();

    // CVWidget size
    int PointsWidgetHeight, PointsWidgetWidth;

private:
    Ui::pointsselectwindow *ui;
};

#endif // POINTSSELECTWINDOW_H
