#ifndef __TOOLS_COLOR_PRINT_HPP__
#define __TOOLS_COLOR_PRINT_HPP__
#include <stdio.h>
#include <string.h>
/*
 * Make screen print colorful :)
 * Author: Jiarong Lin
 * Related link: 
 * [1] https://en.wikipedia.org/wiki/ANSI_escape_code 
 * [2] https://github.com/caffedrine/CPP_Utils/blob/597fe5200be87fa1db2d2aa5d8a07c3bc32a66cd/Utils/include/AnsiColors.h
 * 
*/

const std::string _tools_color_printf_version = "V1.0";
const std::string _tools_color_printf_info = "[Init]: Add macros, add scope_color()";
using std::cout;
using std::endl;
// clang-format off
#ifdef EMPTY_ANSI_COLORS
    #define ANSI_COLOR_RED ""
    #define ANSI_COLOR_RED_BOLD ""
    #define ANSI_COLOR_GREEN ""
    #define ANSI_COLOR_GREEN_BOLD ""
    #define ANSI_COLOR_YELLOW ""
    #define ANSI_COLOR_YELLOW_BOLD ""
    #define ANSI_COLOR_BLUE ""
    #define ANSI_COLOR_BLUE_BOLD ""
    #define ANSI_COLOR_MAGENTA ""
    #define ANSI_COLOR_MAGENTA_BOLD ""
#else
    #define ANSI_COLOR_RED "\x1b[0;31m"
    #define ANSI_COLOR_RED_BOLD "\x1b[1;31m"
    #define ANSI_COLOR_RED_BG "\x1b[0;41m"

    #define ANSI_COLOR_GREEN "\x1b[0;32m"
    #define ANSI_COLOR_GREEN_BOLD "\x1b[1;32m"
    #define ANSI_COLOR_GREEN_BG "\x1b[0;42m"

    #define ANSI_COLOR_YELLOW "\x1b[0;33m"
    #define ANSI_COLOR_YELLOW_BOLD "\x1b[1;33m"
    #define ANSI_COLOR_YELLOW_BG "\x1b[0;43m"

    #define ANSI_COLOR_BLUE "\x1b[0;34m"
    #define ANSI_COLOR_BLUE_BOLD "\x1b[1;34m"
    #define ANSI_COLOR_BLUE_BG "\x1b[0;44m"

    #define ANSI_COLOR_MAGENTA "\x1b[0;35m"
    #define ANSI_COLOR_MAGENTA_BOLD "\x1b[1;35m"
    #define ANSI_COLOR_MAGENTA_BG "\x1b[0;45m"

    #define ANSI_COLOR_CYAN "\x1b[0;36m"
    #define ANSI_COLOR_CYAN_BOLD "\x1b[1;36m"
    #define ANSI_COLOR_CYAN_BG "\x1b[0;46m"

    #define ANSI_COLOR_WHITE "\x1b[0;37m"
    #define ANSI_COLOR_WHITE_BOLD "\x1b[1;37m"
    #define ANSI_COLOR_WHITE_BG "\x1b[0;47m"

    #define ANSI_COLOR_BLACK "\x1b[0;30m"
    #define ANSI_COLOR_BLACK_BOLD "\x1b[1;30m"
    #define ANSI_COLOR_BLACK_BG "\x1b[0;40m"

    #define ANSI_COLOR_RESET "\x1b[0m"
#endif
// clang-format on

struct _Scope_color
{
    _Scope_color( const char * color )
    {
        cout << color;
    }

    ~_Scope_color()
    {
        cout << ANSI_COLOR_RESET;
    }
};

#define scope_color(a) _Scope_color _scope(a);

inline int demo_test_color_printf()
{
    int i, j, n;

    for ( i = 0; i < 11; i++ )
    {
        for ( j = 0; j < 10; j++ )
        {
            n = 10 * i + j;
            if ( n > 108 )
                break;
            printf( "\033[%dm %3d\033[m", n, n );
        }
        printf( "\n" );
    }
    return ( 0 );
};
#endif