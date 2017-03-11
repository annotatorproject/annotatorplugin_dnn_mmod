#ifndef MMOD_H
#define MMOD_H

#include <annotator/plugins/plugin.h>
#include "widget.h"

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/image_processing.h>
#include <QtCore/QObject>
#include <QtCore/QtPlugin>
#include <QtGui/QIcon>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <thread>
#include <vector>

using std::shared_ptr;
using namespace AnnotatorLib;

namespace AnnotatorLib {
class Session;
}

namespace Annotator {
namespace Plugins {

class MMOD : public Plugin {
  Q_OBJECT
  Q_PLUGIN_METADATA(IID "annotator.mmod" FILE "mmod.json")
  Q_INTERFACES(Annotator::Plugin)

  // A 5x5 conv layer that does 2x downsampling
  template <long num_filters, typename SUBNET>
  using con5d = dlib::con<num_filters, 5, 5, 2, 2, SUBNET>;
  // A 3x3 conv layer that doesn't do any downsampling
  template <long num_filters, typename SUBNET>
  using con3 = dlib::con<num_filters, 3, 3, 1, 1, SUBNET>;

  // Now we can define the 8x downsampling block in terms of conv5d blocks.  We
  // also use relu and batch normalization in the standard way.
  template <typename SUBNET>
  using downsampler = dlib::relu<dlib::bn_con<
      con5d<32, dlib::relu<dlib::bn_con<
                    con5d<32, dlib::relu<dlib::bn_con<con5d<32, SUBNET>>>>>>>>>;

  // The rest of the network will be 3x3 conv layers with batch normalization
  // and
  // relu.  So we define the 3x3 block we will use here.
  template <typename SUBNET>
  using rcon3 = dlib::relu<dlib::bn_con<con3<32, SUBNET>>>;

  using net_type = dlib::loss_mmod<
      dlib::con<1, 6, 6, 1, 1,
                rcon3<rcon3<rcon3<downsampler<
                    dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

 public:
  MMOD();
  ~MMOD();
  QString getName() override;
  QWidget *getWidget() override;
  bool setFrame(shared_ptr<Frame> frame, cv::Mat image) override;
  void setObject(shared_ptr<Object> object) override;
  shared_ptr<Object> getObject() const override;
  void setLastAnnotation(shared_ptr<Annotation>) override;
  std::vector<shared_ptr<Commands::Command>> getCommands() override;

  void train();
  void stop();
  void getImagesTrain();

  void loadNet(std::string file);
  void saveNet(std::string file);

 protected:
  void trainWorker();
  cv::Rect findObject();
  QPixmap getImgCrop(shared_ptr<AnnotatorLib::Annotation> annotation,
                     int size) const;
  cv::Mat getImg(shared_ptr<AnnotatorLib::Annotation> annotation) const;

  cv::Mat frameImg;
  shared_ptr<Annotation> lastAnnotation = nullptr;
  shared_ptr<Object> object = nullptr;

  Widget widget;
  std::thread trainThread;

  shared_ptr<Frame> frame = nullptr;
  shared_ptr<Frame> lastFrame = nullptr;

  std::vector<dlib::matrix<dlib::rgb_pixel>> images_train;
  std::vector<std::vector<dlib::mmod_rect>> boxes_train;

  net_type net;

  bool stopTraining = false;
};
}
}

#endif  // MMOD_H
