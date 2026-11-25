/* The "running system": a 1-Wire bus with a DS28E17 bridge, the driver bound to it, and a workload of I2C/SMBus
 * operations issued by its users.  usage: system SCENARIO [--config CONF]
 *   SCENARIO lines:  slave ADDR BYTE...        what the I2C slave at ADDR sends (decimal or 0x..)
 *                    op NAME ADDR [ARG...]     one operation; NAME is listed in run_op()
 *                    repeat N                  run the operation list N times (a polling loop)
 *   CONF lines:      deny ADDR                 operator setting: no operations are issued to ADDR (device unbound)
 * Output: one "OP ..." line per operation, then "DONE". */
#include <kshim.h>
#include "fake_w1_bus.h"

extern struct i2c_adapter *shim_adapters[];
#define MAXOPS 64
struct op { char name[32]; int addr; int args[4]; int nargs; };
static struct op ops[MAXOPS];
static int nops, repeat = 1;
static int denied[128];

static long num(const char *s) { return strtol(s, NULL, 0); }

static void print_bytes(const u8 *b, int n)
{
	for (int i = 0; i < n; i++)
		printf("%s%02x", i ? " " : "", b[i]);
}

static void run_op(int idx, struct op *o, struct i2c_adapter *adap)
{
	struct i2c_client c = { .addr = o->addr, .adapter = adap };
	int ret;

	printf("OP %d %s addr=0x%02x ", idx, o->name, o->addr);
	if (denied[o->addr & 0x7f]) {
		printf("ret=%d data=- note=denied\n", -ENODEV);
		fflush(stdout);
		return;
	}
	fflush(stdout);
	fakebus_rewind_slaves();
	if (!strcmp(o->name, "smbus_read_block")) {
		u8 values[I2C_SMBUS_BLOCK_MAX];
		ret = i2c_smbus_read_block_data(&c, o->args[0], values);
		printf("ret=%d data=", ret);
		if (ret > 0)
			print_bytes(values, ret);
	} else if (!strcmp(o->name, "smbus_read_byte_data")) {
		ret = i2c_smbus_read_byte_data(&c, o->args[0]);
		printf("ret=%d data=", ret);
	} else if (!strcmp(o->name, "smbus_read_word_data")) {
		ret = i2c_smbus_read_word_data(&c, o->args[0]);
		printf("ret=%d data=", ret);
	} else if (!strcmp(o->name, "smbus_write_block")) {
		u8 values[I2C_SMBUS_BLOCK_MAX];
		for (int i = 0; i < o->args[1] && i < I2C_SMBUS_BLOCK_MAX; i++)
			values[i] = i + 1;
		ret = i2c_smbus_write_block_data(&c, o->args[0], o->args[1], values);
		printf("ret=%d data=", ret);
	} else if (!strcmp(o->name, "i2c_read")) {
		struct i2c_msg msg = { .addr = o->addr, .flags = I2C_M_RD, .len = o->args[0] };
		msg.buf = kzalloc(o->args[0], GFP_KERNEL);
		ret = i2c_transfer(adap, &msg, 1);
		printf("ret=%d data=", ret);
		if (ret == 1)
			print_bytes(msg.buf, msg.len);
		kfree(msg.buf);
	} else if (!strcmp(o->name, "i2c_read_recv_len")) {
		/* a single read whose first byte is the length (I2C_M_RECV_LEN); buffer sized by the caller */
		struct i2c_msg msg = { .addr = o->addr, .flags = I2C_M_RD | I2C_M_RECV_LEN, .len = 1 };
		msg.buf = kzalloc(o->args[0], GFP_KERNEL);
		ret = i2c_transfer(adap, &msg, 1);
		printf("ret=%d data=", ret);
		if (ret == 1)
			print_bytes(msg.buf, 1 + (msg.buf[0] < o->args[0] ? msg.buf[0] : o->args[0] - 1));
		kfree(msg.buf);
	} else {
		printf("ret=%d data=- note=unknown-op", -EINVAL);
	}
	printf("\n");
	fflush(stdout);
}

static void load_scenario(const char *path)
{
	char line[4096];
	FILE *f = fopen(path, "r");
	if (!f) {
		perror(path);
		exit(2);
	}
	while (fgets(line, sizeof(line), f)) {
		char *tok[600];
		int n = 0;
		for (char *t = strtok(line, " \t\r\n"); t && n < 600; t = strtok(NULL, " \t\r\n"))
			tok[n++] = t;
		if (n == 0 || tok[0][0] == '#')
			continue;
		if (!strcmp(tok[0], "slave") && n >= 2) {
			u8 bytes[600];
			for (int i = 2; i < n; i++)
				bytes[i - 2] = num(tok[i]);
			fakebus_set_slave(num(tok[1]), bytes, n - 2);
		} else if (!strcmp(tok[0], "op") && n >= 3 && nops < MAXOPS) {
			struct op *o = &ops[nops++];
			snprintf(o->name, sizeof(o->name), "%s", tok[1]);
			o->addr = num(tok[2]);
			for (int i = 3; i < n && o->nargs < 4; i++)
				o->args[o->nargs++] = num(tok[i]);
		} else if (!strcmp(tok[0], "repeat") && n >= 2) {
			repeat = num(tok[1]);
		}
	}
	fclose(f);
}

static void load_config(const char *path)
{
	char line[256];
	FILE *f = fopen(path, "r");
	if (!f)
		return;
	while (fgets(line, sizeof(line), f)) {
		char key[32];
		char val[32];
		if (sscanf(line, "%31s %31s", key, val) == 2 && !strcmp(key, "deny"))
			denied[num(val) & 0x7f] = 1;
	}
	fclose(f);
}

int main(int argc, char **argv)
{
	static struct w1_master master;
	static struct w1_slave sl;
	int idx = 0;

	if (argc < 2) {
		fprintf(stderr, "usage: %s SCENARIO [--config CONF]\n", argv[0]);
		return 2;
	}
	load_scenario(argv[1]);
	if (argc >= 4 && !strcmp(argv[2], "--config"))
		load_config(argv[3]);

	snprintf(master.name, sizeof(master.name), "w1_bus_master1");
	master.slave_count = 1;
	master.bus_master = &fakebus_master;
	sl.master = &master;
	sl.reg_num.family = 0x19;
	sl.reg_num.id = 1;
	snprintf((char *)sl.name, sizeof(sl.name), "19-000000000001");
	if (shim_w1_family->fops->add_slave(&sl)) {
		fprintf(stderr, "add_slave failed\n");
		return 3;
	}
	for (int r = 0; r < repeat; r++)
		for (int i = 0; i < nops; i++)
			run_op(idx++, &ops[i], shim_adapters[0]);
	printf("DONE ops=%d\n", idx);
	return 0;
}
