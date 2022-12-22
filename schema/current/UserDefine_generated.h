// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_USERDEFINE_MNN_H_
#define FLATBUFFERS_GENERATED_USERDEFINE_MNN_H_


#include "Tensor_generated.h"
#include "Type_generated.h"

namespace MNN {

struct TensorConvertInfo;
struct TensorConvertInfoT;

struct GridSample;
struct GridSampleT;

struct ImageProcessParam;
struct ImageProcessParamT;

inline const flatbuffers::TypeTable *TensorConvertInfoTypeTable();

inline const flatbuffers::TypeTable *GridSampleTypeTable();

inline const flatbuffers::TypeTable *ImageProcessParamTypeTable();

enum SampleMode {
  SampleMode_BILINEAR = 0,
  SampleMode_NEAREST = 1,
  SampleMode_MIN = SampleMode_BILINEAR,
  SampleMode_MAX = SampleMode_NEAREST
};

inline const SampleMode (&EnumValuesSampleMode())[2] {
  static const SampleMode values[] = {
    SampleMode_BILINEAR,
    SampleMode_NEAREST
  };
  return values;
}

inline const char * const *EnumNamesSampleMode() {
  static const char * const names[] = {
    "BILINEAR",
    "NEAREST",
    nullptr
  };
  return names;
}

inline const char *EnumNameSampleMode(SampleMode e) {
  if (e < SampleMode_BILINEAR || e > SampleMode_NEAREST) return "";
  const size_t index = static_cast<int>(e);
  return EnumNamesSampleMode()[index];
}

enum BorderMode {
  BorderMode_ZEROS = 0,
  BorderMode_CLAMP = 1,
  BorderMode_REFLECTION = 2,
  BorderMode_MIN = BorderMode_ZEROS,
  BorderMode_MAX = BorderMode_REFLECTION
};

inline const BorderMode (&EnumValuesBorderMode())[3] {
  static const BorderMode values[] = {
    BorderMode_ZEROS,
    BorderMode_CLAMP,
    BorderMode_REFLECTION
  };
  return values;
}

inline const char * const *EnumNamesBorderMode() {
  static const char * const names[] = {
    "ZEROS",
    "CLAMP",
    "REFLECTION",
    nullptr
  };
  return names;
}

inline const char *EnumNameBorderMode(BorderMode e) {
  if (e < BorderMode_ZEROS || e > BorderMode_REFLECTION) return "";
  const size_t index = static_cast<int>(e);
  return EnumNamesBorderMode()[index];
}

enum ImageFormatType {
  ImageFormatType_RGBA = 0,
  ImageFormatType_RGB = 1,
  ImageFormatType_BGR = 2,
  ImageFormatType_GRAY = 3,
  ImageFormatType_BGRA = 4,
  ImageFormatType_YCrCb = 5,
  ImageFormatType_YUV = 6,
  ImageFormatType_HSV = 7,
  ImageFormatType_XYZ = 8,
  ImageFormatType_BGR555 = 9,
  ImageFormatType_BGR565 = 10,
  ImageFormatType_YUV_NV21 = 11,
  ImageFormatType_YUV_NV12 = 12,
  ImageFormatType_YUV_I420 = 13,
  ImageFormatType_HSV_FULL = 14,
  ImageFormatType_MIN = ImageFormatType_RGBA,
  ImageFormatType_MAX = ImageFormatType_HSV_FULL
};

inline const ImageFormatType (&EnumValuesImageFormatType())[15] {
  static const ImageFormatType values[] = {
    ImageFormatType_RGBA,
    ImageFormatType_RGB,
    ImageFormatType_BGR,
    ImageFormatType_GRAY,
    ImageFormatType_BGRA,
    ImageFormatType_YCrCb,
    ImageFormatType_YUV,
    ImageFormatType_HSV,
    ImageFormatType_XYZ,
    ImageFormatType_BGR555,
    ImageFormatType_BGR565,
    ImageFormatType_YUV_NV21,
    ImageFormatType_YUV_NV12,
    ImageFormatType_YUV_I420,
    ImageFormatType_HSV_FULL
  };
  return values;
}

inline const char * const *EnumNamesImageFormatType() {
  static const char * const names[] = {
    "RGBA",
    "RGB",
    "BGR",
    "GRAY",
    "BGRA",
    "YCrCb",
    "YUV",
    "HSV",
    "XYZ",
    "BGR555",
    "BGR565",
    "YUV_NV21",
    "YUV_NV12",
    "YUV_I420",
    "HSV_FULL",
    nullptr
  };
  return names;
}

inline const char *EnumNameImageFormatType(ImageFormatType e) {
  if (e < ImageFormatType_RGBA || e > ImageFormatType_HSV_FULL) return "";
  const size_t index = static_cast<int>(e);
  return EnumNamesImageFormatType()[index];
}

enum FilterType {
  FilterType_NEAREST = 0,
  FilterType_BILINEAR = 1,
  FilterType_BICUBIC = 2,
  FilterType_MIN = FilterType_NEAREST,
  FilterType_MAX = FilterType_BICUBIC
};

inline const FilterType (&EnumValuesFilterType())[3] {
  static const FilterType values[] = {
    FilterType_NEAREST,
    FilterType_BILINEAR,
    FilterType_BICUBIC
  };
  return values;
}

inline const char * const *EnumNamesFilterType() {
  static const char * const names[] = {
    "NEAREST",
    "BILINEAR",
    "BICUBIC",
    nullptr
  };
  return names;
}

inline const char *EnumNameFilterType(FilterType e) {
  if (e < FilterType_NEAREST || e > FilterType_BICUBIC) return "";
  const size_t index = static_cast<int>(e);
  return EnumNamesFilterType()[index];
}

enum WrapType {
  WrapType_CLAMP_TO_EDGE = 0,
  WrapType_ZERO = 1,
  WrapType_REPEAT = 2,
  WrapType_MIN = WrapType_CLAMP_TO_EDGE,
  WrapType_MAX = WrapType_REPEAT
};

inline const WrapType (&EnumValuesWrapType())[3] {
  static const WrapType values[] = {
    WrapType_CLAMP_TO_EDGE,
    WrapType_ZERO,
    WrapType_REPEAT
  };
  return values;
}

inline const char * const *EnumNamesWrapType() {
  static const char * const names[] = {
    "CLAMP_TO_EDGE",
    "ZERO",
    "REPEAT",
    nullptr
  };
  return names;
}

inline const char *EnumNameWrapType(WrapType e) {
  if (e < WrapType_CLAMP_TO_EDGE || e > WrapType_REPEAT) return "";
  const size_t index = static_cast<int>(e);
  return EnumNamesWrapType()[index];
}

struct TensorConvertInfoT : public flatbuffers::NativeTable {
  typedef TensorConvertInfo TableType;
  MNN_DATA_FORMAT source;
  MNN_DATA_FORMAT dest;
  TensorConvertInfoT()
      : source(MNN_DATA_FORMAT_NCHW),
        dest(MNN_DATA_FORMAT_NCHW) {
  }
};

struct TensorConvertInfo FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef TensorConvertInfoT NativeTableType;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return TensorConvertInfoTypeTable();
  }
  MNN_DATA_FORMAT source() const {
    return static_cast<MNN_DATA_FORMAT>(GetField<int8_t>(4, 0));
  }
  MNN_DATA_FORMAT dest() const {
    return static_cast<MNN_DATA_FORMAT>(GetField<int8_t>(6, 0));
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int8_t>(verifier, 4) &&
           VerifyField<int8_t>(verifier, 6) &&
           verifier.EndTable();
  }
  TensorConvertInfoT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(TensorConvertInfoT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<TensorConvertInfo> Pack(flatbuffers::FlatBufferBuilder &_fbb, const TensorConvertInfoT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct TensorConvertInfoBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_source(MNN_DATA_FORMAT source) {
    fbb_.AddElement<int8_t>(4, static_cast<int8_t>(source), 0);
  }
  void add_dest(MNN_DATA_FORMAT dest) {
    fbb_.AddElement<int8_t>(6, static_cast<int8_t>(dest), 0);
  }
  explicit TensorConvertInfoBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  TensorConvertInfoBuilder &operator=(const TensorConvertInfoBuilder &);
  flatbuffers::Offset<TensorConvertInfo> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<TensorConvertInfo>(end);
    return o;
  }
};

inline flatbuffers::Offset<TensorConvertInfo> CreateTensorConvertInfo(
    flatbuffers::FlatBufferBuilder &_fbb,
    MNN_DATA_FORMAT source = MNN_DATA_FORMAT_NCHW,
    MNN_DATA_FORMAT dest = MNN_DATA_FORMAT_NCHW) {
  TensorConvertInfoBuilder builder_(_fbb);
  builder_.add_dest(dest);
  builder_.add_source(source);
  return builder_.Finish();
}

flatbuffers::Offset<TensorConvertInfo> CreateTensorConvertInfo(flatbuffers::FlatBufferBuilder &_fbb, const TensorConvertInfoT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

struct GridSampleT : public flatbuffers::NativeTable {
  typedef GridSample TableType;
  SampleMode mode;
  BorderMode paddingMode;
  bool alignCorners;
  GridSampleT()
      : mode(SampleMode_BILINEAR),
        paddingMode(BorderMode_ZEROS),
        alignCorners(false) {
  }
};

struct GridSample FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef GridSampleT NativeTableType;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return GridSampleTypeTable();
  }
  SampleMode mode() const {
    return static_cast<SampleMode>(GetField<int8_t>(4, 0));
  }
  BorderMode paddingMode() const {
    return static_cast<BorderMode>(GetField<int8_t>(6, 0));
  }
  bool alignCorners() const {
    return GetField<uint8_t>(8, 0) != 0;
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int8_t>(verifier, 4) &&
           VerifyField<int8_t>(verifier, 6) &&
           VerifyField<uint8_t>(verifier, 8) &&
           verifier.EndTable();
  }
  GridSampleT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(GridSampleT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<GridSample> Pack(flatbuffers::FlatBufferBuilder &_fbb, const GridSampleT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct GridSampleBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_mode(SampleMode mode) {
    fbb_.AddElement<int8_t>(4, static_cast<int8_t>(mode), 0);
  }
  void add_paddingMode(BorderMode paddingMode) {
    fbb_.AddElement<int8_t>(6, static_cast<int8_t>(paddingMode), 0);
  }
  void add_alignCorners(bool alignCorners) {
    fbb_.AddElement<uint8_t>(8, static_cast<uint8_t>(alignCorners), 0);
  }
  explicit GridSampleBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  GridSampleBuilder &operator=(const GridSampleBuilder &);
  flatbuffers::Offset<GridSample> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<GridSample>(end);
    return o;
  }
};

inline flatbuffers::Offset<GridSample> CreateGridSample(
    flatbuffers::FlatBufferBuilder &_fbb,
    SampleMode mode = SampleMode_BILINEAR,
    BorderMode paddingMode = BorderMode_ZEROS,
    bool alignCorners = false) {
  GridSampleBuilder builder_(_fbb);
  builder_.add_alignCorners(alignCorners);
  builder_.add_paddingMode(paddingMode);
  builder_.add_mode(mode);
  return builder_.Finish();
}

flatbuffers::Offset<GridSample> CreateGridSample(flatbuffers::FlatBufferBuilder &_fbb, const GridSampleT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

struct ImageProcessParamT : public flatbuffers::NativeTable {
  typedef ImageProcessParam TableType;
  FilterType filterType;
  ImageFormatType sourceFormat;
  ImageFormatType destFormat;
  WrapType wrap;
  std::vector<float> mean;
  std::vector<float> normal;
  std::vector<float> transform;
  int8_t paddingValue;
  std::vector<int32_t> shape;
  DataType outputType;
  bool draw;
  ImageProcessParamT()
      : filterType(FilterType_NEAREST),
        sourceFormat(ImageFormatType_RGBA),
        destFormat(ImageFormatType_RGBA),
        wrap(WrapType_CLAMP_TO_EDGE),
        paddingValue(0),
        outputType(DataType_DT_INVALID),
        draw(false) {
  }
};

struct ImageProcessParam FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef ImageProcessParamT NativeTableType;
  static const flatbuffers::TypeTable *MiniReflectTypeTable() {
    return ImageProcessParamTypeTable();
  }
  FilterType filterType() const {
    return static_cast<FilterType>(GetField<int8_t>(4, 0));
  }
  ImageFormatType sourceFormat() const {
    return static_cast<ImageFormatType>(GetField<int32_t>(6, 0));
  }
  ImageFormatType destFormat() const {
    return static_cast<ImageFormatType>(GetField<int32_t>(8, 0));
  }
  WrapType wrap() const {
    return static_cast<WrapType>(GetField<int8_t>(10, 0));
  }
  const flatbuffers::Vector<float> *mean() const {
    return GetPointer<const flatbuffers::Vector<float> *>(12);
  }
  const flatbuffers::Vector<float> *normal() const {
    return GetPointer<const flatbuffers::Vector<float> *>(14);
  }
  const flatbuffers::Vector<float> *transform() const {
    return GetPointer<const flatbuffers::Vector<float> *>(16);
  }
  int8_t paddingValue() const {
    return GetField<int8_t>(18, 0);
  }
  const flatbuffers::Vector<int32_t> *shape() const {
    return GetPointer<const flatbuffers::Vector<int32_t> *>(20);
  }
  DataType outputType() const {
    return static_cast<DataType>(GetField<int32_t>(22, 0));
  }
  bool draw() const {
    return GetField<uint8_t>(24, 0) != 0;
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int8_t>(verifier, 4) &&
           VerifyField<int32_t>(verifier, 6) &&
           VerifyField<int32_t>(verifier, 8) &&
           VerifyField<int8_t>(verifier, 10) &&
           VerifyOffset(verifier, 12) &&
           verifier.VerifyVector(mean()) &&
           VerifyOffset(verifier, 14) &&
           verifier.VerifyVector(normal()) &&
           VerifyOffset(verifier, 16) &&
           verifier.VerifyVector(transform()) &&
           VerifyField<int8_t>(verifier, 18) &&
           VerifyOffset(verifier, 20) &&
           verifier.VerifyVector(shape()) &&
           VerifyField<int32_t>(verifier, 22) &&
           VerifyField<uint8_t>(verifier, 24) &&
           verifier.EndTable();
  }
  ImageProcessParamT *UnPack(const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  void UnPackTo(ImageProcessParamT *_o, const flatbuffers::resolver_function_t *_resolver = nullptr) const;
  static flatbuffers::Offset<ImageProcessParam> Pack(flatbuffers::FlatBufferBuilder &_fbb, const ImageProcessParamT* _o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);
};

struct ImageProcessParamBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_filterType(FilterType filterType) {
    fbb_.AddElement<int8_t>(4, static_cast<int8_t>(filterType), 0);
  }
  void add_sourceFormat(ImageFormatType sourceFormat) {
    fbb_.AddElement<int32_t>(6, static_cast<int32_t>(sourceFormat), 0);
  }
  void add_destFormat(ImageFormatType destFormat) {
    fbb_.AddElement<int32_t>(8, static_cast<int32_t>(destFormat), 0);
  }
  void add_wrap(WrapType wrap) {
    fbb_.AddElement<int8_t>(10, static_cast<int8_t>(wrap), 0);
  }
  void add_mean(flatbuffers::Offset<flatbuffers::Vector<float>> mean) {
    fbb_.AddOffset(12, mean);
  }
  void add_normal(flatbuffers::Offset<flatbuffers::Vector<float>> normal) {
    fbb_.AddOffset(14, normal);
  }
  void add_transform(flatbuffers::Offset<flatbuffers::Vector<float>> transform) {
    fbb_.AddOffset(16, transform);
  }
  void add_paddingValue(int8_t paddingValue) {
    fbb_.AddElement<int8_t>(18, paddingValue, 0);
  }
  void add_shape(flatbuffers::Offset<flatbuffers::Vector<int32_t>> shape) {
    fbb_.AddOffset(20, shape);
  }
  void add_outputType(DataType outputType) {
    fbb_.AddElement<int32_t>(22, static_cast<int32_t>(outputType), 0);
  }
  void add_draw(bool draw) {
    fbb_.AddElement<uint8_t>(24, static_cast<uint8_t>(draw), 0);
  }
  explicit ImageProcessParamBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ImageProcessParamBuilder &operator=(const ImageProcessParamBuilder &);
  flatbuffers::Offset<ImageProcessParam> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<ImageProcessParam>(end);
    return o;
  }
};

inline flatbuffers::Offset<ImageProcessParam> CreateImageProcessParam(
    flatbuffers::FlatBufferBuilder &_fbb,
    FilterType filterType = FilterType_NEAREST,
    ImageFormatType sourceFormat = ImageFormatType_RGBA,
    ImageFormatType destFormat = ImageFormatType_RGBA,
    WrapType wrap = WrapType_CLAMP_TO_EDGE,
    flatbuffers::Offset<flatbuffers::Vector<float>> mean = 0,
    flatbuffers::Offset<flatbuffers::Vector<float>> normal = 0,
    flatbuffers::Offset<flatbuffers::Vector<float>> transform = 0,
    int8_t paddingValue = 0,
    flatbuffers::Offset<flatbuffers::Vector<int32_t>> shape = 0,
    DataType outputType = DataType_DT_INVALID,
    bool draw = false) {
  ImageProcessParamBuilder builder_(_fbb);
  builder_.add_outputType(outputType);
  builder_.add_shape(shape);
  builder_.add_transform(transform);
  builder_.add_normal(normal);
  builder_.add_mean(mean);
  builder_.add_destFormat(destFormat);
  builder_.add_sourceFormat(sourceFormat);
  builder_.add_draw(draw);
  builder_.add_paddingValue(paddingValue);
  builder_.add_wrap(wrap);
  builder_.add_filterType(filterType);
  return builder_.Finish();
}

flatbuffers::Offset<ImageProcessParam> CreateImageProcessParam(flatbuffers::FlatBufferBuilder &_fbb, const ImageProcessParamT *_o, const flatbuffers::rehasher_function_t *_rehasher = nullptr);

inline TensorConvertInfoT *TensorConvertInfo::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = new TensorConvertInfoT();
  UnPackTo(_o, _resolver);
  return _o;
}

inline void TensorConvertInfo::UnPackTo(TensorConvertInfoT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = source(); _o->source = _e; };
  { auto _e = dest(); _o->dest = _e; };
}

inline flatbuffers::Offset<TensorConvertInfo> TensorConvertInfo::Pack(flatbuffers::FlatBufferBuilder &_fbb, const TensorConvertInfoT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateTensorConvertInfo(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<TensorConvertInfo> CreateTensorConvertInfo(flatbuffers::FlatBufferBuilder &_fbb, const TensorConvertInfoT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const TensorConvertInfoT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _source = _o->source;
  auto _dest = _o->dest;
  return MNN::CreateTensorConvertInfo(
      _fbb,
      _source,
      _dest);
}

inline GridSampleT *GridSample::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = new GridSampleT();
  UnPackTo(_o, _resolver);
  return _o;
}

inline void GridSample::UnPackTo(GridSampleT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = mode(); _o->mode = _e; };
  { auto _e = paddingMode(); _o->paddingMode = _e; };
  { auto _e = alignCorners(); _o->alignCorners = _e; };
}

inline flatbuffers::Offset<GridSample> GridSample::Pack(flatbuffers::FlatBufferBuilder &_fbb, const GridSampleT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateGridSample(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<GridSample> CreateGridSample(flatbuffers::FlatBufferBuilder &_fbb, const GridSampleT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const GridSampleT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _mode = _o->mode;
  auto _paddingMode = _o->paddingMode;
  auto _alignCorners = _o->alignCorners;
  return MNN::CreateGridSample(
      _fbb,
      _mode,
      _paddingMode,
      _alignCorners);
}

inline ImageProcessParamT *ImageProcessParam::UnPack(const flatbuffers::resolver_function_t *_resolver) const {
  auto _o = new ImageProcessParamT();
  UnPackTo(_o, _resolver);
  return _o;
}

inline void ImageProcessParam::UnPackTo(ImageProcessParamT *_o, const flatbuffers::resolver_function_t *_resolver) const {
  (void)_o;
  (void)_resolver;
  { auto _e = filterType(); _o->filterType = _e; };
  { auto _e = sourceFormat(); _o->sourceFormat = _e; };
  { auto _e = destFormat(); _o->destFormat = _e; };
  { auto _e = wrap(); _o->wrap = _e; };
  { auto _e = mean(); if (_e) { _o->mean.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->mean[_i] = _e->Get(_i); } } };
  { auto _e = normal(); if (_e) { _o->normal.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->normal[_i] = _e->Get(_i); } } };
  { auto _e = transform(); if (_e) { _o->transform.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->transform[_i] = _e->Get(_i); } } };
  { auto _e = paddingValue(); _o->paddingValue = _e; };
  { auto _e = shape(); if (_e) { _o->shape.resize(_e->size()); for (flatbuffers::uoffset_t _i = 0; _i < _e->size(); _i++) { _o->shape[_i] = _e->Get(_i); } } };
  { auto _e = outputType(); _o->outputType = _e; };
  { auto _e = draw(); _o->draw = _e; };
}

inline flatbuffers::Offset<ImageProcessParam> ImageProcessParam::Pack(flatbuffers::FlatBufferBuilder &_fbb, const ImageProcessParamT* _o, const flatbuffers::rehasher_function_t *_rehasher) {
  return CreateImageProcessParam(_fbb, _o, _rehasher);
}

inline flatbuffers::Offset<ImageProcessParam> CreateImageProcessParam(flatbuffers::FlatBufferBuilder &_fbb, const ImageProcessParamT *_o, const flatbuffers::rehasher_function_t *_rehasher) {
  (void)_rehasher;
  (void)_o;
  struct _VectorArgs { flatbuffers::FlatBufferBuilder *__fbb; const ImageProcessParamT* __o; const flatbuffers::rehasher_function_t *__rehasher; } _va = { &_fbb, _o, _rehasher}; (void)_va;
  auto _filterType = _o->filterType;
  auto _sourceFormat = _o->sourceFormat;
  auto _destFormat = _o->destFormat;
  auto _wrap = _o->wrap;
  auto _mean = _o->mean.size() ? _fbb.CreateVector(_o->mean) : 0;
  auto _normal = _o->normal.size() ? _fbb.CreateVector(_o->normal) : 0;
  auto _transform = _o->transform.size() ? _fbb.CreateVector(_o->transform) : 0;
  auto _paddingValue = _o->paddingValue;
  auto _shape = _o->shape.size() ? _fbb.CreateVector(_o->shape) : 0;
  auto _outputType = _o->outputType;
  auto _draw = _o->draw;
  return MNN::CreateImageProcessParam(
      _fbb,
      _filterType,
      _sourceFormat,
      _destFormat,
      _wrap,
      _mean,
      _normal,
      _transform,
      _paddingValue,
      _shape,
      _outputType,
      _draw);
}

inline const flatbuffers::TypeTable *SampleModeTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    SampleModeTypeTable
  };
  static const char * const names[] = {
    "BILINEAR",
    "NEAREST"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_ENUM, 2, type_codes, type_refs, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *BorderModeTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    BorderModeTypeTable
  };
  static const char * const names[] = {
    "ZEROS",
    "CLAMP",
    "REFLECTION"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_ENUM, 3, type_codes, type_refs, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *ImageFormatTypeTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 },
    { flatbuffers::ET_INT, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    ImageFormatTypeTypeTable
  };
  static const char * const names[] = {
    "RGBA",
    "RGB",
    "BGR",
    "GRAY",
    "BGRA",
    "YCrCb",
    "YUV",
    "HSV",
    "XYZ",
    "BGR555",
    "BGR565",
    "YUV_NV21",
    "YUV_NV12",
    "YUV_I420",
    "HSV_FULL"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_ENUM, 15, type_codes, type_refs, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *FilterTypeTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    FilterTypeTypeTable
  };
  static const char * const names[] = {
    "NEAREST",
    "BILINEAR",
    "BICUBIC"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_ENUM, 3, type_codes, type_refs, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *WrapTypeTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    WrapTypeTypeTable
  };
  static const char * const names[] = {
    "CLAMP_TO_EDGE",
    "ZERO",
    "REPEAT"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_ENUM, 3, type_codes, type_refs, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *TensorConvertInfoTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 0 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    MNN_DATA_FORMATTypeTable
  };
  static const char * const names[] = {
    "source",
    "dest"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 2, type_codes, type_refs, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *GridSampleTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_CHAR, 0, 1 },
    { flatbuffers::ET_BOOL, 0, -1 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    SampleModeTypeTable,
    BorderModeTypeTable
  };
  static const char * const names[] = {
    "mode",
    "paddingMode",
    "alignCorners"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 3, type_codes, type_refs, nullptr, names
  };
  return &tt;
}

inline const flatbuffers::TypeTable *ImageProcessParamTypeTable() {
  static const flatbuffers::TypeCode type_codes[] = {
    { flatbuffers::ET_CHAR, 0, 0 },
    { flatbuffers::ET_INT, 0, 1 },
    { flatbuffers::ET_INT, 0, 1 },
    { flatbuffers::ET_CHAR, 0, 2 },
    { flatbuffers::ET_FLOAT, 1, -1 },
    { flatbuffers::ET_FLOAT, 1, -1 },
    { flatbuffers::ET_FLOAT, 1, -1 },
    { flatbuffers::ET_CHAR, 0, -1 },
    { flatbuffers::ET_INT, 1, -1 },
    { flatbuffers::ET_INT, 0, 3 },
    { flatbuffers::ET_BOOL, 0, -1 }
  };
  static const flatbuffers::TypeFunction type_refs[] = {
    FilterTypeTypeTable,
    ImageFormatTypeTypeTable,
    WrapTypeTypeTable,
    DataTypeTypeTable
  };
  static const char * const names[] = {
    "filterType",
    "sourceFormat",
    "destFormat",
    "wrap",
    "mean",
    "normal",
    "transform",
    "paddingValue",
    "shape",
    "outputType",
    "draw"
  };
  static const flatbuffers::TypeTable tt = {
    flatbuffers::ST_TABLE, 11, type_codes, type_refs, nullptr, names
  };
  return &tt;
}

}  // namespace MNN

#endif  // FLATBUFFERS_GENERATED_USERDEFINE_MNN_H_
