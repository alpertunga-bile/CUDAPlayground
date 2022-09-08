#pragma once

#include <chrono>
#include <iostream>

class Timer
{
public:
	void Start()
	{
		startTime = std::chrono::system_clock::now();
	}

	void End()
	{
		endTime = std::chrono::system_clock::now();
		elapsedTime = endTime - startTime;
		miliseconds = elapsedTime.count();
		seconds = miliseconds / 1000.0;
		minutes = seconds / 60.0;
	}

	void Print(const char* functionName)
	{
		std::cout << functionName << " : " << minutes << " min | " << seconds << " s | " << miliseconds << " ms\n";
	}

	double GetMiliseconds() { return miliseconds; }
	double GetSeconds() { return seconds; }
	double GetMinutes() { return minutes; }

private:
	std::chrono::time_point<std::chrono::system_clock> startTime, endTime;
	std::chrono::duration<double> elapsedTime;
	double miliseconds;
	double seconds;
	double minutes;
};