#ifndef __DACE_WORMHOLE_HOST_H
#define __DACE_WORMHOLE_HOST_H

namespace dace { namespace wormhole {

  struct Context {
    tt::tt_metal::Device* device;
    tt::tt_metal::Program program;

    Context() {}
    ~Context() {}
  };

}}

#endif
