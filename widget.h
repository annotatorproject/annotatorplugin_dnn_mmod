#ifndef WIDGET_H
#define WIDGET_H

#include <QPixmap>
#include <QWidget>

namespace Ui {
class Widget;
}

namespace Annotator {
namespace Plugins {
class MMOD;
}
}

class Widget : public QWidget {
  Q_OBJECT

 public:
  explicit Widget(QWidget *parent = 0);
  ~Widget();
  void setObjectPixmap(QPixmap pixmap);
  void setMMOD(Annotator::Plugins::MMOD *mmod);
  void setProgress(int percent);

 private slots:

  void on_trainButton_clicked();

  void on_saveButton_clicked();

  void on_loadButton_clicked();

 private:
  Ui::Widget *ui;
  Annotator::Plugins::MMOD *mmod;
  bool training = false;
};

#endif  // WIDGET_H
