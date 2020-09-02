#include "gtest/gtest.h"

#include "main.cpp"


namespace {

TEST(CountTailZeroTest, All) {
    EXPECT_EQ(3, add(1, 2));
    EXPECT_EQ(5, add(3, 2));
    EXPECT_EQ(1, add(-1, 2));
}

}
