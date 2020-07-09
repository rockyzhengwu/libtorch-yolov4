#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>
#include <map>
#include <typeinfo>
#include <algorithm>


typedef std::map<std::string, std::string> Block;


class Config
{
public:
  Config(const char* cfg_file);
  Config();

  std::vector<Block> blocks_;
  void load_cfg();
  std::string  cfg_file_;

  static int get_int_from_block(Block &block, std::string key, int default_value);
  static int split(std::string s, std::vector<std::string> &res, std::string delimiter=",");
  static int split(std::string& str, std::vector<int>& ret_, std::string delimiter= ",");

  static void trim(std::string &s);
  static void ltrim(std::string &s);
  static void rtrim(std::string &s);
  static std::string get_string_from_block(Block &block, std::string key, std::string default_value);

};
#endif