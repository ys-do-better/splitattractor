/* User-space stand-ins for the kernel APIs that the code under test (kernel/) uses.
 * Nothing here is under test: it only lets the upstream code compile and run as a normal process. */
#ifndef KSHIM_H
#define KSHIM_H
#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
typedef u8 __u8;
typedef u16 __u16;
typedef u32 __u32;

#define __stringify_1(x...) #x
#define __stringify(x...) __stringify_1(x)
#define MODULE_LICENSE(x)
#define MODULE_AUTHOR(x)
#define MODULE_DESCRIPTION(x)
#define MODULE_ALIAS(x)
#define MODULE_PARM_DESC(a, b)
#define module_param_named(name, value, type, perm)
#define EXPORT_SYMBOL(x)
#define EXPORT_SYMBOL_GPL(x)
#define THIS_MODULE NULL
struct module;

#define container_of(ptr, type, member) ((type *)((char *)(ptr) - offsetof(type, member)))
#define GFP_KERNEL 0
static inline void *kzalloc(size_t n, int gfp) { (void)gfp; return calloc(1, n); }
static inline void kfree(const void *p) { free((void *)p); }

struct device { const char *init_name; struct device *parent; };
static inline void *devm_kzalloc(struct device *d, size_t n, int gfp) { (void)d; (void)gfp; return calloc(1, n); }
static inline void devm_kfree(struct device *d, void *p) { (void)d; free(p); }
#define dev_warn(dev, fmt, ...) fprintf(stderr, "kernel: warn: " fmt, ##__VA_ARGS__)
#define dev_err(dev, fmt, ...) fprintf(stderr, "kernel: err: " fmt, ##__VA_ARGS__)
#define dev_info(dev, fmt, ...) fprintf(stderr, "kernel: info: " fmt, ##__VA_ARGS__)
#define dev_dbg(dev, fmt, ...) do { } while (0)
#define pr_debug(fmt, ...) do { } while (0)
#define scnprintf snprintf
#define sysfs_emit(buf, fmt, ...) sprintf(buf, fmt, ##__VA_ARGS__)

struct attribute { const char *name; unsigned short mode; };
struct device_attribute {
	struct attribute attr;
	ssize_t (*show)(struct device *, struct device_attribute *, char *);
	ssize_t (*store)(struct device *, struct device_attribute *, const char *, size_t);
};
#define DEVICE_ATTR_RW(_name) \
	struct device_attribute dev_attr_##_name = { .attr = { .name = #_name, .mode = 0644 }, \
		.show = _name##_show, .store = _name##_store }
struct attribute_group { const char *name; struct attribute **attrs; };

static inline void usleep_range(unsigned long a, unsigned long b) { (void)a; (void)b; }
static inline void udelay(unsigned long a) { (void)a; }
static inline void msleep(unsigned int a) { (void)a; }
#define might_sleep() do { } while (0)
#define local_irq_save(f) ((void)(f))
#define local_irq_restore(f) ((void)(f))

struct mutex { int locked; };
static inline void mutex_lock(struct mutex *m) { m->locked = 1; }
static inline void mutex_unlock(struct mutex *m) { m->locked = 0; }
struct list_head { struct list_head *next, *prev; };
typedef struct { int counter; } atomic_t;
#define le64_to_cpu(x) (x)

u16 crc16(u16 crc, const u8 *buffer, size_t len);

/* ---- I2C (values from include/uapi/linux/i2c.h and include/linux/i2c.h) ---- */
#define I2C_M_RD 0x0001
#define I2C_M_TEN 0x0010
#define I2C_M_DMA_SAFE 0x0200
#define I2C_M_RECV_LEN 0x0400
#define I2C_CLIENT_PEC 0x04
#define I2C_CLIENT_SCCB 0x9000
#define I2C_SMBUS_BLOCK_MAX 32
#define I2C_SMBUS_READ 1
#define I2C_SMBUS_WRITE 0
#define I2C_SMBUS_QUICK 0
#define I2C_SMBUS_BYTE 1
#define I2C_SMBUS_BYTE_DATA 2
#define I2C_SMBUS_WORD_DATA 3
#define I2C_SMBUS_PROC_CALL 4
#define I2C_SMBUS_BLOCK_DATA 5
#define I2C_SMBUS_I2C_BLOCK_BROKEN 6
#define I2C_SMBUS_BLOCK_PROC_CALL 7
#define I2C_SMBUS_I2C_BLOCK_DATA 8
#define I2C_FUNC_I2C 0x00000001
#define I2C_FUNC_SMBUS_PEC 0x00000008
#define I2C_FUNC_SMBUS_READ_BYTE 0x00020000
#define I2C_FUNC_SMBUS_WRITE_BYTE 0x00040000
#define I2C_FUNC_SMBUS_READ_BYTE_DATA 0x00080000
#define I2C_FUNC_SMBUS_WRITE_BYTE_DATA 0x00100000
#define I2C_FUNC_SMBUS_READ_WORD_DATA 0x00200000
#define I2C_FUNC_SMBUS_WRITE_WORD_DATA 0x00400000
#define I2C_FUNC_SMBUS_PROC_CALL 0x00800000
#define I2C_FUNC_SMBUS_WRITE_BLOCK_DATA 0x02000000
#define I2C_FUNC_SMBUS_READ_I2C_BLOCK 0x04000000
#define I2C_FUNC_SMBUS_WRITE_I2C_BLOCK 0x08000000
#define I2C_FUNC_SMBUS_BYTE (I2C_FUNC_SMBUS_READ_BYTE | I2C_FUNC_SMBUS_WRITE_BYTE)
#define I2C_FUNC_SMBUS_BYTE_DATA (I2C_FUNC_SMBUS_READ_BYTE_DATA | I2C_FUNC_SMBUS_WRITE_BYTE_DATA)
#define I2C_FUNC_SMBUS_WORD_DATA (I2C_FUNC_SMBUS_READ_WORD_DATA | I2C_FUNC_SMBUS_WRITE_WORD_DATA)
#define I2C_FUNC_SMBUS_I2C_BLOCK (I2C_FUNC_SMBUS_READ_I2C_BLOCK | I2C_FUNC_SMBUS_WRITE_I2C_BLOCK)

union i2c_smbus_data { u8 byte; u16 word; u8 block[I2C_SMBUS_BLOCK_MAX + 2]; };
struct i2c_msg { u16 addr; u16 flags; u16 len; u8 *buf; };
struct i2c_adapter;
struct i2c_algorithm {
	int (*master_xfer)(struct i2c_adapter *adap, struct i2c_msg *msgs, int num);
	int (*master_xfer_atomic)(struct i2c_adapter *adap, struct i2c_msg *msgs, int num);
	int (*smbus_xfer)(struct i2c_adapter *adap, u16 addr, unsigned short flags, char read_write,
			  u8 command, int size, union i2c_smbus_data *data);
	int (*smbus_xfer_atomic)(struct i2c_adapter *adap, u16 addr, unsigned short flags, char read_write,
				 u8 command, int size, union i2c_smbus_data *data);
	u32 (*functionality)(struct i2c_adapter *adap);
};
struct i2c_adapter_quirks {
	u64 flags;
	int max_num_msgs;
	u16 max_write_len;
	u16 max_read_len;
	u16 max_comb_1st_msg_len;
	u16 max_comb_2nd_msg_len;
};
struct i2c_adapter {
	struct module *owner;
	unsigned int class;
	const struct i2c_algorithm *algo;
	void *algo_data;
	int timeout;
	int retries;
	struct device dev;
	int nr;
	char name[48];
	const struct i2c_adapter_quirks *quirks;
};
struct i2c_client { unsigned short flags; unsigned short addr; char name[20]; struct i2c_adapter *adapter; };
int i2c_add_adapter(struct i2c_adapter *adap);
void i2c_del_adapter(struct i2c_adapter *adap);
int __i2c_transfer(struct i2c_adapter *adap, struct i2c_msg *msgs, int num);
int i2c_transfer(struct i2c_adapter *adap, struct i2c_msg *msgs, int num);
static inline u8 i2c_8bit_addr_from_msg(const struct i2c_msg *msg)
{
	return (msg->addr << 1) | (msg->flags & I2C_M_RD);
}
#define str_read_write(v) ((v) ? "read" : "write")
s32 i2c_smbus_xfer(struct i2c_adapter *adapter, u16 addr, unsigned short flags, char read_write,
		   u8 command, int protocol, union i2c_smbus_data *data);
s32 __i2c_smbus_xfer(struct i2c_adapter *adapter, u16 addr, unsigned short flags, char read_write,
		     u8 command, int protocol, union i2c_smbus_data *data);
s32 i2c_smbus_read_byte_data(const struct i2c_client *client, u8 command);
s32 i2c_smbus_read_word_data(const struct i2c_client *client, u8 command);
s32 i2c_smbus_read_block_data(const struct i2c_client *client, u8 command, u8 *values);
s32 i2c_smbus_write_block_data(const struct i2c_client *client, u8 command, u8 length, const u8 *values);

/* ---- 1-Wire (include/linux/w1.h) ---- */
#define W1_MAXNAMELEN 32
#define W1_SEARCH 0xF0
#define W1_ALARM_SEARCH 0xEC
#define W1_CONVERT_TEMP 0x44
#define W1_SKIP_ROM 0xCC
#define W1_COPY_SCRATCHPAD 0x48
#define W1_WRITE_SCRATCHPAD 0x4E
#define W1_READ_SCRATCHPAD 0xBE
#define W1_READ_ROM 0x33
#define W1_READ_PSUPPLY 0xB4
#define W1_MATCH_ROM 0x55
#define W1_RESUME_CMD 0xA5
struct w1_reg_num { u64 family:8, id:48, crc:8; };
struct w1_bus_master {
	void *data;
	u8 (*read_bit)(void *);
	void (*write_bit)(void *, u8);
	u8 (*touch_bit)(void *, u8);
	u8 (*read_byte)(void *);
	void (*write_byte)(void *, u8);
	u8 (*read_block)(void *, u8 *, int);
	void (*write_block)(void *, const u8 *, int);
	u8 (*triplet)(void *, u8);
	u8 (*reset_bus)(void *);
	u8 (*set_pullup)(void *, int);
};
struct w1_master {
	char name[W1_MAXNAMELEN];
	int slave_count;
	int pullup_duration;
	int enable_pullup;
	struct mutex bus_mutex;
	struct w1_bus_master *bus_master;
};
struct w1_slave {
	struct module *owner;
	unsigned char name[W1_MAXNAMELEN];
	struct w1_reg_num reg_num;
	struct w1_master *master;
	struct w1_family *family;
	void *family_data;
	struct device dev;
};
struct w1_family_ops {
	int (*add_slave)(struct w1_slave *sl);
	void (*remove_slave)(struct w1_slave *sl);
	const struct attribute_group **groups;
};
struct w1_family { u8 fid; const struct w1_family_ops *fops; };
#define dev_to_w1_slave(_dev) container_of((_dev), struct w1_slave, dev)
u8 w1_touch_bit(struct w1_master *dev, int bit);
void w1_write_8(struct w1_master *dev, u8 byte);
u8 w1_read_8(struct w1_master *dev);
void w1_write_block(struct w1_master *dev, const u8 *buf, int len);
u8 w1_read_block(struct w1_master *dev, u8 *buf, int len);
int w1_reset_bus(struct w1_master *dev);
int w1_reset_select_slave(struct w1_slave *sl);
int w1_reset_resume_command(struct w1_master *dev);
extern struct w1_family *shim_w1_family;
#define module_w1_family(__w1_family) struct w1_family *shim_w1_family = &(__w1_family)
#endif
