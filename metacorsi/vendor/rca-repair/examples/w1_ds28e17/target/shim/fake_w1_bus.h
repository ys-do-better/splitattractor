#ifndef FAKE_W1_BUS_H
#define FAKE_W1_BUS_H
extern struct w1_bus_master fakebus_master;
void fakebus_set_slave(int addr, const u8 *bytes, int n);
void fakebus_rewind_slaves(void);
#endif
