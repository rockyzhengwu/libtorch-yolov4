#include "config.h"
#include <iostream>
#include <fstream>
#include <stdio.h>

Config::Config(const char *cfg_file){
  cfg_file_=cfg_file;
  load_cfg();
}
Config::Config(){

}

void Config::load_cfg()
{
  std::string line;
  std::ifstream fs(cfg_file_);
  if (!fs)
  {
    std::cout << "faild to load config file \n";
    exit(0);
  }
  while (std::getline(fs, line))
  {
    trim(line);
    if (line.length() == 0)
      continue;
    if(line.substr(0, 1)== "["){
      Block layer_option ;
      layer_option["type"] = line.substr(1, line.length()-2);
      blocks_.push_back(layer_option);
    } else if(line.substr(0, 1) == "#" ){
      continue;
    }
    else{
      std::vector<std::string> values ;
      split(line, values, "=");
      if(values.size()!=2){
        std::cout << "invalid config: " << line << "" << values.size() << "\n";
        continue;
      }
      blocks_.back().insert(std::pair<std::string, std::string>(values[0], values[1]));
    }
  }
}


int Config::get_int_from_block(Block &block, std::string key, int default_value){
  if(block.find(key)==block.end()){
    return default_value;
  }
  return std::stoi(block.at(key));
}

std::string Config::get_string_from_block(Block &block, std::string key, std::string default_value){
  if(block.find(key)==block.end()){
    return default_value;
  }
  return block.at(key);
}


void Config::ltrim(std::string &s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
            return !std::isspace(ch);
          }));
}

void Config::rtrim(std::string &s)
{
  s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
            return !std::isspace(ch);
          }).base(),
          s.end());
}

void Config::trim(std::string &s)
{
  ltrim(s);
  rtrim(s);
}

int Config::split(std::string s, std::vector<std::string> &res, std::string delimiter)
{
  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos)
  {
    token = s.substr(0, pos);
    trim(token);
    if (token.length())
    {
      res.push_back(token);
    }
    s.erase(0, pos + delimiter.length());
  }
  if (s.length())
  {
    res.push_back(s);
  }
  return 0;
}

int Config::split(std::string& s, std::vector<int> &res, std::string delimiter){
  std::vector<std::string> tmp;
	split(s, tmp, delimiter);
	for(int i = 0; i < tmp.size(); i++)
	{
		res.push_back(std::stoi(tmp[i]));
	}
}
