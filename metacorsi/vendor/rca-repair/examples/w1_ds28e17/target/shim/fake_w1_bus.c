/* An emulated 1-Wire bus master with one DS28E17 (family 0x19) and I2C slaves behind it.
 * The DS28E17 side follows the command set in the driver (and the Maxim datasheet): the master writes a function
 * command, the bridge runs the I2C transaction against the slave model, then returns status and data bytes.
 * Each I2C slave replies from a programmed byte stream: this is how a test states what the peripheral sends. */
#include <kshim.h>
#include "fake_w1_bus.h"

#define QMAX 4096
struct slave_model { int present; u8 stream[QMAX]; int len; int pos; };
static struct slave_model slaves[128];
static u8 outq[QMAX];
static int outq_len, outq_pos;
static u8 inb[QMAX];
static int in_len;
static int rom_done;
static u8 config_byte = 1;

void fakebus_set_slave(int addr, const u8 *bytes, int n)
{
	struct slave_model *s = &slaves[addr & 0x7f];
	s->present = 1;
	s->len = n < QMAX ? n : QMAX;
	memcpy(s->stream, bytes, s->len);
	s->pos = 0;
}

void fakebus_rewind_slaves(void)
{
	for (int i = 0; i < 128; i++)
		slaves[i].pos = 0;
}

static u8 slave_next(int addr)
{
	struct slave_model *s = &slaves[addr & 0x7f];
	return s->pos < s->len ? s->stream[s->pos++] : 0xff;
}

static void out(u8 b)
{
	if (outq_len < QMAX)
		outq[outq_len++] = b;
}

/* status byte: 0x02 = I2C address not acknowledged */
static void run_read(int addr, int count)
{
	if (!slaves[addr & 0x7f].present) {
		out(0x02);
		return;
	}
	out(0x00);
	for (int i = 0; i < count; i++)
		out(slave_next(addr));
}

static void execute_if_complete(void)
{
	u8 cmd = inb[0];
	int need, wcount;

	switch (cmd) {
	case 0x2D: /* write, then read, with stop: cmd addr wcount wdata[wcount] rcount crc crc */
		if (in_len < 3)
			return;
		wcount = inb[2];
		need = 3 + wcount + 1 + 2;
		if (in_len < need)
			return;
		if (!slaves[(inb[1] >> 1) & 0x7f].present) {
			out(0x02);
			out(0x00);
		} else {
			int rcount = inb[3 + wcount];
			out(0x00);
			out(0x00);
			for (int i = 0; i < rcount; i++)
				out(slave_next(inb[1] >> 1));
		}
		break;
	case 0x87: /* read with stop: cmd addr count crc crc */
		if (in_len < 5)
			return;
		run_read(inb[1] >> 1, inb[2]);
		break;
	case 0x4B: case 0x5A: /* write with/without stop: cmd addr count data[count] crc crc */
		if (in_len < 3 || in_len < 3 + inb[2] + 2)
			return;
		out(slaves[(inb[1] >> 1) & 0x7f].present ? 0x00 : 0x02);
		out(0x00);
		break;
	case 0x69: case 0x78: /* write data only: cmd count data[count] crc crc */
		if (in_len < 2 || in_len < 2 + inb[1] + 2)
			return;
		out(0x00);
		out(0x00);
		break;
	case 0xD2: /* write configuration: cmd value */
		if (in_len < 2)
			return;
		config_byte = inb[1];
		break;
	case 0xE1: /* read configuration */
		out(config_byte);
		break;
	default:
		break;
	}
	in_len = 0;
}

static u8 fb_reset_bus(void *data)
{
	(void)data;
	in_len = 0;
	rom_done = 0;
	outq_len = outq_pos = 0;
	return 0; /* presence pulse seen */
}

static u8 fb_touch_bit(void *data, u8 bit)
{
	(void)data;
	(void)bit;
	return 0; /* the bridge is never busy */
}

static u8 fb_read_byte(void *data)
{
	(void)data;
	return outq_pos < outq_len ? outq[outq_pos++] : 0xff;
}

static void fb_write_byte(void *data, u8 b)
{
	(void)data;
	if (!rom_done) {
		inb[in_len++] = b;
		if (inb[0] == W1_SKIP_ROM || inb[0] == W1_RESUME_CMD || (inb[0] == W1_MATCH_ROM && in_len == 9)) {
			rom_done = 1;
			in_len = 0;
		}
		return;
	}
	if (in_len < QMAX)
		inb[in_len++] = b;
	execute_if_complete();
}

struct w1_bus_master fakebus_master = {
	.touch_bit = fb_touch_bit,
	.read_byte = fb_read_byte,
	.write_byte = fb_write_byte,
	.reset_bus = fb_reset_bus,
};
