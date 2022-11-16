
// how to deal with the __device__ CUDA stuff that bindgen will not be able to
// understand
class ChannelDev {
private:
  int id;
  volatile int *doorbell;

  uint8_t *buff;
  uint8_t *buff_end;

  // head/tail pointers
  uint8_t *volatile buff_write_head_ptr;
  uint8_t *volatile buff_write_tail_ptr;

public:
  ChannelDev() {}

  /*     __device__ __forceinline__ void push(void* packet, uint32_t nbytes) {*/
  /*         assert(nbytes != 0);*/

  /*         uint8_t* curr_ptr = NULL;*/

  /*         while (curr_ptr == NULL) {*/
  /*             curr_ptr =*/
  /*                 (uint8_t*)atomicAdd((ULL*)&buff_write_head_ptr,
   * (ULL)nbytes);*/

  /*             // if the current position plus nbytes is after buff_end, the*/
  /*             // buffer is full.*/
  /*             // Many warps could find condition true, but only the first
   * warp*/
  /*             // will find true the condition after. */
  /*             if (curr_ptr + nbytes > buff_end) {*/
  /*                 // I am the first warp that found the buffer full and*/
  /*                 // I am the one responsible for flushing the buffer out */
  /*                 if (curr_ptr <= buff_end) {*/
  /*                     // wait until everyone completed to write */
  /*                     while (buff_write_tail_ptr != curr_ptr) {*/
  /*                     }*/

  /*                     // flush buffer */
  /*                     flush();*/
  /*                 } else {*/
  /*                     // waiting for buffer to flush */
  /*                     while (buff_write_head_ptr > buff_end) {*/
  /*                     }*/
  /*                 }*/
  /*                 curr_ptr = NULL;*/
  /*             }*/
  /*         }*/

  /*         memcpy(curr_ptr, packet, nbytes);*/
  /*         atomicAdd((ULL*)&buff_write_tail_ptr, (ULL)nbytes);*/
  /*     }*/

  /*     __device__ __forceinline__ void flush() {*/
  /*         uint32_t nbytes = (uint32_t)(buff_write_tail_ptr - buff);*/
  /*         // printf("FLUSH CHANNEL#%d: buffer bytes %d\n", id, nbytes);*/
  /*         if (nbytes == 0) {*/
  /*             return;*/
  /*         }*/

  /*         // make sure everything is visible in memory */
  /*         __threadfence_system();*/

  /*         assert(*doorbell == 0);*/
  /*         // notify current buffer has something */
  /*         *doorbell = nbytes;*/

  /*         // wait for host to release the doorbell */
  /*         while (*doorbell != 0)*/
  /*             ;*/

  /*         // reset head/tail */
  /*         buff_write_tail_ptr = buff;*/
  /*         __threadfence();*/
  /*         buff_write_head_ptr = buff;*/

  /*         // printf("FLUSH CHANNEL#%d: DONE\n", id);*/
  /*     }*/

private:
  // called by the ChannelHost init
  void init(int id, int *h_doorbell, int buff_size);
  /* void init(int id, int* h_doorbell, int buff_size) { */
  /*     CUDA_SAFECALL( */
  /*         cudaHostGetDevicePointer((void**)&doorbell, (void*)h_doorbell, 0));
   */

  /* // allocate large buffer */
  /* #ifdef USE_ASYNC_STREAM*/
  /*         CUDA_SAFECALL(cudaMalloc((void**)&buff, buff_size));*/
  /* #else*/
  /*         CUDA_SAFECALL(cudaMallocManaged((void**)&buff, buff_size));*/
  /* #endif*/
  /*         buff_write_head_ptr = buff;*/
  /*         buff_write_tail_ptr = buff;*/
  /*         buff_end = buff + buff_size;*/
  /*         this->id = id;*/
  /*     }*/

  friend class ChannelHost;
};

class ChannelHost {
private:
  volatile int *doorbell;

  /* cudaStream_t *stream; */
  CUstream stream;
  /* cudaStream_t stream; */
  /* CUstream_st stream; */
  /* CUstream_st stream; */
  ChannelDev *ch_dev;

  /* pointers to device buffer */
  uint8_t *dev_buff_read_head;
  uint8_t *dev_buff;

  /* receiving thread */
  pthread_t thread;
  volatile bool thread_started;

public:
  int id;
  int buff_size;

public:
  ChannelHost() {}

  void init(int id, int buff_size, ChannelDev *ch_dev,
            void *(*thread_fun)(void *), void *args = NULL);

  /* void init(int id, int buff_size, ChannelDev* ch_dev, */
  /*           void* (*thread_fun)(void*), void* args = NULL) { */
  /*     this->buff_size = buff_size; */
  /*     this->id = id; */
  /*     /1* get device properties *1/ */
  /*     cudaDeviceProp prop; */
  /*     int device = 0; */
  /*     cudaGetDeviceProperties(&prop, device); */
  /*     if (prop.canMapHostMemory == 0) { */
  /*         CUDA_SAFECALL(cudaSetDeviceFlags(cudaDeviceMapHost)); */
  /*     } */

  /* #ifdef USE_ASYNC_STREAM */
  /*     /1* create stream that will read memory with highest possible priority
   * *1/ */
  /*     int priority_high, priority_low; */
  /*     CUDA_SAFECALL( */
  /*         cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
   */
  /*     CUDA_SAFECALL(cudaStreamCreateWithPriority( */
  /*         &stream, cudaStreamNonBlocking, priority_high)); */
  /* #endif */

  /*     /1* create doorbell *1/ */
  /*     CUDA_SAFECALL( */
  /*         cudaHostAlloc((void**)&doorbell, sizeof(int),
   * cudaHostAllocMapped)); */
  /*     /1* set doorbell to zero *1/ */
  /*     *doorbell = 0; */

  /*     /1* initialize device channel *1/ */
  /*     this->ch_dev = ch_dev; */
  /*     ch_dev->init(id, (int*)doorbell, buff_size); */

  /*     dev_buff = ch_dev->buff; */
  /*     dev_buff_read_head = dev_buff; */
  /*     if (thread_fun != NULL) { */
  /*         thread_started = true; */
  /*         pthread_create(&thread, NULL, (void* (*)(void*))thread_fun, args);
   */
  /*     } else { */
  /*         thread_started = false; */
  /*     } */
  /* } */

  /* when used in nvbit we don't want to dealloc because
   * when modules are unloaded the driver automatically
   * deallocates CUDA malloc, so further deallocs done
   * here will result in errors */
  void destroy(bool dealloc);

  /* void destroy(bool dealloc) { */
  /*     if (thread_started) { */
  /*         thread_started = false; */
  /*         pthread_join(thread, NULL); */
  /*     } */
  /*     if (dealloc) { */
  /* #ifdef USE_ASYNC_STREAM */
  /*         CUDA_SAFECALL(cudaStreamDestroy(stream)); */
  /* #endif */
  /*         CUDA_SAFECALL(cudaFree((int*)doorbell)); */
  /*         CUDA_SAFECALL(cudaFree(ch_dev->buff)); */
  /*     } */
  /* } */

  bool is_active();
  /* bool is_active() { return thread_started; } */

  uint32_t recv(void *buff, uint32_t max_buff_size);
  /* uint32_t recv(void* buff, uint32_t max_buff_size) { */
  /*     assert(max_buff_size > 0); */
  /*     assert(doorbell != NULL); */
  /*     uint32_t buff_nbytes = *doorbell; */
  /*     if (buff_nbytes == 0) { */
  /*         return 0; */
  /*     } */
  /*     int nbytes = buff_nbytes; */

  /*     if (buff_nbytes > max_buff_size) { */
  /*         nbytes = max_buff_size; */
  /*     } */
  /* #ifdef USE_ASYNC_STREAM */
  /*     CUDA_SAFECALL(cudaMemcpyAsync(buff, dev_buff_read_head, nbytes, */
  /*                                   cudaMemcpyDeviceToHost, stream)); */
  /*     CUDA_SAFECALL(cudaStreamSynchronize(stream)); */
  /* #else */
  /*     memcpy(buff, dev_buff_read_head, nbytes); */
  /* #endif */
  /*     int bytes_left = buff_nbytes - nbytes; */
  /*     assert(bytes_left >= 0); */
  /*     if (bytes_left > 0) { */
  /*         dev_buff_read_head += nbytes; */
  /*     } else { */
  /*         dev_buff_read_head = dev_buff; */
  /*     } */

  /*     *doorbell = bytes_left; */
  /*     // printf("HOST RECEIVED nbytes %d - bytes left %d\n", nbytes, */
  /*     // bytes_left); */
  /*     return nbytes; */
  /* } */

  pthread_t get_thread();
  /* pthread_t get_thread() { return thread; } */

  friend class MultiChannelHost;
};
